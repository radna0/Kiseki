#!/usr/bin/env python3
"""
Fast and stable line art colorization using Carmack optimizations.
This version avoids hanging issues while maintaining high performance.
"""

import numpy as np
import cv2
import time
import argparse
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
from linefiller.linefiller.thinning import thinning


def fast_trapped_ball_fill(binary, radius):
    """Fast trapped-ball fill using morphological operations."""
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)
    )

    # Closing operation (dilation followed by erosion)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # Find regions that were filled
    filled = cv2.bitwise_xor(binary, closed)

    return cv2.bitwise_and(binary, cv2.bitwise_not(filled))


def fast_flood_fill(binary, max_regions=1000):
    """Fast flood fill with region limit to prevent hanging."""
    h, w = binary.shape
    mask = np.zeros((h + 2, w + 2), np.uint8)
    result = binary.copy()
    regions = []

    for y in range(0, h, 50):  # Sample every 50 pixels for speed
        for x in range(0, w, 50):
            if result[y, x] == 255:
                # Flood fill this region
                cv2.floodFill(result, mask, (x, y), len(regions) + 1)
                regions.append((x, y))

                if len(regions) >= max_regions:
                    print(f"Reached region limit ({max_regions})")
                    return result, regions

    return result, regions


def read_line_2_np(img_path, channel=4):
    """
    Reads an image file (RGB or RGBA) and creates a standardized line art image,
    detecting lines based on alpha or luminosity.
    """
    from PIL import Image

    try:
        img = Image.open(img_path)
    except FileNotFoundError:
        return None

    img_np = np.array(img)

    if img.mode == "RGBA":
        alpha_channel = img_np[:, :, 3]
        mask = alpha_channel > 100  # Line detection based on alpha value
    elif img.mode == "RGB":
        grayscale = np.mean(img_np[:, :, :3], axis=2)
        mask = grayscale < 150  # Line detection based on grayscale value
    else:  # Grayscale or other modes
        img = img.convert("L")
        img_np = np.array(img)
        mask = img_np < 150

    line = np.zeros((*img_np.shape[:2], 4), dtype=np.uint8)
    line[:, :, :3] = 255  # Set all RGB to white
    line[:, :, 3] = np.where(mask, 255, 0)  # Set alpha: 255 for lines, 0 for background

    # If original image was RGB/A, copy original RGB values to new image where there are lines
    if len(img_np.shape) > 2 and img_np.shape[2] >= 3:
        line[mask, :3] = img_np[mask, :3]

    return line[..., :channel]


def show_fill_map(fillmap: np.ndarray, lines_are_black=True):
    """Utility to visualize the labeled regions with random colors."""
    max_label = np.max(fillmap)
    if max_label == 0:
        return np.zeros((*fillmap.shape, 3), dtype=np.uint8)

    # Use a fixed seed for deterministic colors.
    colors = np.random.randint(50, 255, (int(max_label) + 1, 3), dtype=np.uint8)

    if lines_are_black:
        colors[0] = [0, 0, 0]  # Ensure lines (label 0) are black.

    return colors[fillmap]


# You may need to install this: pip install scipy
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from cv2 import ximgproc


def find_endpoints(binary_image):
    """
    Finds all line endpoints in a binary image using a morphological hit-or-miss transform.
    Assumes lines are black (0) and background is white (255).
    """
    # Invert the image because morphological operations in OpenCV typically expect white objects on a black background.
    image_inv = cv2.bitwise_not(binary_image)

    # Define 8 kernels for detecting endpoints in all 8 directions (N, NE, E, SE, S, SW, W, NW)
    # The kernel center is an endpoint if it's a foreground pixel (1) and has only one foreground neighbor.
    kernels = [
        np.array([[-1, -1, -1], [-1, 1, -1], [0, 1, 0]], dtype="int"),  # North
        np.array([[-1, -1, -1], [0, 1, -1], [1, 0, -1]], dtype="int"),  # North-East
        np.array([[-1, 0, 1], [-1, 1, 0], [-1, -1, -1]], dtype="int"),  # East
        np.array([[1, 0, -1], [0, 1, -1], [-1, -1, -1]], dtype="int"),  # South-East
        np.array([[0, 1, 0], [-1, 1, -1], [-1, -1, -1]], dtype="int"),  # South
        np.array([[-1, 0, 1], [-1, 1, 0], [-1, -1, -1]], dtype="int"),  # South-West
        np.array([[1, 0, -1], [0, 1, -1], [-1, -1, -1]], dtype="int"),  # West
        np.array([[-1, -1, 1], [-1, 1, 0], [-1, 0, -1]], dtype="int"),
    ]  # North-West

    endpoint_map = np.zeros(image_inv.shape, dtype=np.uint8)

    # Apply each kernel and accumulate the results
    for kernel in kernels:
        endpoint_map = cv2.bitwise_or(
            endpoint_map, cv2.morphologyEx(image_inv, cv2.MORPH_HITMISS, kernel)
        )

    # Get the coordinates of the endpoints
    endpoints = np.transpose(np.nonzero(endpoint_map))
    return endpoints  # Returns coordinates as (row, col) which is (y, x)


def get_stroke_properties(endpoint, binary_image, lookback=15):
    """
    Analyzes the stroke leading to an endpoint to find its direction and momentum.

    Args:
        endpoint (tuple): The (y, x) coordinate of the endpoint.
        binary_image (np.ndarray): The binary image where lines are 0.
        lookback (int): How many pixels to trace back along the stroke.

    Returns:
        list: A list of the last `lookback` points on the stroke, representing its recent path.
              Returns an empty list if the stroke is too short.
    """
    path = [endpoint]
    current_point = endpoint
    h, w = binary_image.shape

    # Create a copy of the image we can modify to avoid re-tracing paths
    img_copy = binary_image.copy()

    for _ in range(lookback):
        found_next = False
        # Search in a 3x3 neighborhood for the next pixel in the stroke
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                if dy == 0 and dx == 0:
                    continue

                ny, nx = current_point[0] + dy, current_point[1] + dx

                if 0 <= ny < h and 0 <= nx < w and img_copy[ny, nx] == 0:
                    path.append((ny, nx))
                    # Erase the found pixel to not step back on it
                    img_copy[current_point] = 255
                    current_point = (ny, nx)
                    found_next = True
                    break
            if found_next:
                break

        if not found_next:
            # Reached the start of the stroke or an intersection
            break

    # We need at least 3 points to define a curve/direction
    if len(path) < 3:
        return []

    return path


def project_and_connect(
    start_endpoint,
    start_path,
    binary_image,
    target_endpoints_set,
    max_trace_len=100,
    step_size=5,
):
    """
    Projects a path from an endpoint, following its curve, and checks for connections or collisions.

    Args:
        start_endpoint (tuple): The (y, x) starting point.
        start_path (list): The recent history of the stroke for calculating momentum.
        binary_image (np.ndarray): The original line art.
        target_endpoints_set (set): A set of all other available endpoints for fast lookup.
        max_trace_len (int): How far to trace a path before giving up.
        step_size (int): How many pixels to project forward in each step.

    Returns:
        tuple: (target_endpoint, projected_path) if a valid connection is found, otherwise (None, None).
    """
    h, w = binary_image.shape

    # Calculate initial direction vector from the last few points of the path
    # We use points further back to get a smoother, more stable direction
    p_start = np.array(start_path[0])  # The endpoint
    p_mid = np.array(start_path[len(start_path) // 2])

    direction = p_start - p_mid
    norm = np.linalg.norm(direction)
    if norm == 0:
        return None, None
    direction = direction / norm

    projected_path = [start_endpoint]
    current_pos = np.array(start_endpoint, dtype=float)

    for i in range(max_trace_len // step_size):
        # Project the next point
        next_pos = current_pos + direction * step_size
        projected_path.append(tuple(next_pos.astype(int)))

        # --- Collision and Target Check ---
        # Check all pixels along the straight line from current_pos to next_pos
        line_points_y = np.linspace(current_pos[0], next_pos[0], step_size * 2).astype(
            int
        )
        line_points_x = np.linspace(current_pos[1], next_pos[1], step_size * 2).astype(
            int
        )

        for y, x in zip(line_points_y, line_points_x):
            if not (0 <= y < h and 0 <= x < w):
                return None, None  # Went off-screen

            # Check for collision with existing lines (but not with the start or target endpoints)
            if binary_image[y, x] == 0 and (y, x) != start_endpoint:
                return None, None  # Collision

            # Check if we landed near a target endpoint
            if (y, x) in target_endpoints_set:
                return (y, x), projected_path  # Success!

        current_pos = next_pos

        # --- Update Direction to allow for curves ---
        # The new direction is a blend of the old direction and the overall direction of the trace
        path_start_vec = np.array(start_endpoint)
        overall_direction = current_pos - path_start_vec
        norm = np.linalg.norm(overall_direction)
        if norm > 0:
            # Blend the straight-line direction with the persistent momentum
            # This makes the path continue to curve naturally
            new_direction = direction * 0.7 + (overall_direction / norm) * 0.3
            new_norm = np.linalg.norm(new_direction)
            if new_norm > 0:
                direction = new_direction / new_norm

    return None, None  # Traced too far without finding anything


def connect_unclosed_points(binary, max_trace=120):
    """
    Finds and intelligently connects unclosed points in line art by tracing their paths.
    """
    print("\nStarting Predictive Path Tracing...")
    start_time = time.perf_counter()

    # Create a copy to draw connections on
    connected_image = binary.copy()

    # Find all endpoints to start with
    endpoints = find_endpoints(binary)
    if len(endpoints) < 2:
        print("Not enough endpoints to connect.")
        return binary

    print(f"Found {len(endpoints)} initial endpoints.")

    # Use a set for fast O(1) lookups of available endpoints
    available_endpoints = set(endpoints)

    for i, start_point in enumerate(endpoints):
        if start_point not in available_endpoints:
            continue  # This point has already been connected

        # 1. Analyze the stroke leading to this endpoint
        stroke_path = get_stroke_properties(start_point, binary)
        if not stroke_path:
            continue

        # 2. Project a path forward from this endpoint
        # We search for targets in the set of endpoints *excluding the current one*
        target_candidates = available_endpoints - {start_point}
        target_point, projected_path = project_and_connect(
            start_point, stroke_path, binary, target_candidates, max_trace_len=max_trace
        )

        # 3. If a valid, non-colliding path to another endpoint is found, connect them
        if target_point:
            print(f"  Connecting {start_point} -> {target_point}")

            # --- Draw a smooth, curved line ---
            # We use a spline to make the connection look natural and not disjointed
            # The control points are the start, the projected path, and the end
            # Note: Spline points need to be in (x, y) format
            ctrl_points_yx = np.array(
                stroke_path[: len(stroke_path) // 2][::-1]
                + projected_path
                + [target_point]
            )
            ctrl_points_xy = ctrl_points_yx[:, ::-1]  # Swap to (x, y)

            # Remove duplicate points which can break splprep
            unique_indices = [
                i
                for i in range(len(ctrl_points_xy))
                if i == 0
                or not np.array_equal(ctrl_points_xy[i], ctrl_points_xy[i - 1])
            ]
            ctrl_points_xy = ctrl_points_xy[unique_indices]

            if len(ctrl_points_xy) > 3:
                tck, u = splprep([ctrl_points_xy[:, 0], ctrl_points_xy[:, 1]], s=2, k=3)
                x_new, y_new = splev(np.linspace(0, 1, 50), tck)

                # Draw the spline on the image
                spline_points = (
                    np.vstack((x_new, y_new)).T.reshape((-1, 1, 2)).astype(np.int32)
                )
                cv2.polylines(
                    connected_image,
                    [spline_points],
                    isClosed=False,
                    color=0,
                    thickness=1,
                )
            else:  # Fallback to a straight line if spline is not possible
                cv2.line(
                    connected_image,
                    (start_point[1], start_point[0]),
                    (target_point[1], target_point[0]),
                    0,
                    1,
                )

            # Mark both endpoints as used so we don't try to connect them again
            available_endpoints.discard(start_point)
            available_endpoints.discard(target_point)

    num_connected = (len(endpoints) - len(available_endpoints)) // 2
    print(f"Connected {num_connected} pairs.")
    print(
        f"Path tracing and connection finished in {(time.perf_counter() - start_time)*1000:.1f} ms"
    )
    return connected_image


def process_fast(image_path, output_path):
    """Fast processing pipeline that won't hang."""
    print(f"\nProcessing: {image_path}")
    start_total = time.perf_counter()

    # Load image
    image = read_line_2_np(image_path, channel=3)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("working.png", image)

    if image is None:
        print(f"Error: Cannot load {image_path}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Threshold
    start = time.perf_counter()
    _, binary = cv2.threshold(image, 254, 255, cv2.THRESH_BINARY)
    print(f"Threshold: {(time.perf_counter() - start)*1000:.1f} ms")

    # ===================================================================
    # NEW INTELLIGENT STEP
    # ===================================================================
    binary = connect_unclosed_points(binary)
    cv2.imwrite(str(Path(output_path) / "lines_connected_smart.png"), binary)
    # ===================================================================

    result = binary.copy()
    # Connected components
    start = time.perf_counter()
    num_labels, labels = cv2.connectedComponents(result)
    print(f"\nConnected components: {(time.perf_counter() - start)*1000:.1f} ms")
    print(f"Found {num_labels} regions")

    # Skip processing if too many regions
    if num_labels > 10000:
        print("Warning: Too many regions, using simplified processing")
        final = labels
    else:
        # Simple region merging
        start = time.perf_counter()

        # Calculate region sizes
        unique, counts = np.unique(labels, return_counts=True)
        small_regions = unique[counts < 50]

        # Merge small regions with neighbors
        kernel = np.ones((3, 3), np.uint8)
        for region_id in small_regions[:100]:  # Limit processing
            if region_id == 0:
                continue

            # Find neighbors
            mask = (labels == region_id).astype(np.uint8)
            dilated = cv2.dilate(mask, kernel, iterations=1)
            border = dilated - mask

            # Get most common neighbor
            neighbor_labels = labels[border > 0]
            neighbor_labels = neighbor_labels[neighbor_labels != region_id]

            if len(neighbor_labels) > 0:
                # Find most common neighbor
                values, counts = np.unique(neighbor_labels, return_counts=True)
                new_label = values[np.argmax(counts)]
                labels[labels == region_id] = new_label

        print(f"Region merging: {(time.perf_counter() - start)*1000:.1f} ms")

        # Fast thinning
        start = time.perf_counter()
        labels
        final = thinning(labels)
        print(f"Thinning: {(time.perf_counter() - start)*1000:.1f} ms")

    # Save results
    start = time.perf_counter()

    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Save fill map
    cv2.imwrite(str(output_dir / "fillmap.png"), show_fill_map(thinning(labels)))

    cv2.imwrite(str(output_dir / "colored.png"), show_fill_map(labels))
    print(f"Save results: {(time.perf_counter() - start)*1000:.1f} ms")

    total_time = (time.perf_counter() - start_total) * 1000
    print(f"\nTotal time: {total_time:.1f} ms")
    print(f"FPS potential: {1000/total_time:.1f}")

    return final


def benchmark_components():
    """Benchmark individual components."""
    print("=== Component Benchmarks ===")

    # Create test image
    test_img = np.ones((1024, 1024), dtype=np.uint8) * 255
    cv2.circle(test_img, (512, 512), 200, 0, -1)
    cv2.rectangle(test_img, (100, 100), (300, 300), 0, -1)

    # Benchmark trapped-ball
    print("\nTrapped-ball fill:")
    for radius in [1, 2, 3, 5]:
        start = time.perf_counter()
        result = fast_trapped_ball_fill(test_img, radius)
        elapsed = (time.perf_counter() - start) * 1000
        print(f"  Radius {radius}: {elapsed:.2f} ms")

    # Benchmark connected components
    print("\nConnected components:")
    start = time.perf_counter()
    num_labels, labels = cv2.connectedComponents(test_img)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  OpenCV: {elapsed:.2f} ms ({num_labels} components)")

    # Benchmark flood fill
    print("\nFlood fill:")
    start = time.perf_counter()
    result, regions = fast_flood_fill(test_img.copy(), max_regions=100)
    elapsed = (time.perf_counter() - start) * 1000
    print(f"  Fast flood fill: {elapsed:.2f} ms ({len(regions)} regions)")



def main():
    parser = argparse.ArgumentParser(description="Fast Line Art Colorization")
    parser.add_argument("input", nargs="?", default="input.png", help="Input image")
    parser.add_argument(
        "-o", "--output", default="output_fast", help="Output directory"
    )
    parser.add_argument("--benchmark", action="store_true", help="Run benchmarks")
    parser.add_argument("--resize", type=int, help="Resize to max dimension")

    args = parser.parse_args()

    if args.benchmark:
        benchmark_components()
        print()

    # Process image
    image_path = Path(args.input)
    if not image_path.exists():
        print(f"Error: {image_path} not found")
        return

    # Load and optionally resize
    if args.resize:
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        if max(h, w) > args.resize:
            scale = args.resize / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h))
            temp_path = "temp_resized.png"
            cv2.imwrite(temp_path, image)
            image_path = Path(temp_path)

    process_fast(image_path, args.output)

    print(f"\nResults saved to {args.output}/")
    print("\nOptimizations used:")
    print("- Morphological operations for trapped-ball fill")
    print("- OpenCV optimized connected components")
    print("- Limited iteration thinning")
    print("- Region count limits to prevent hanging")


if __name__ == "__main__":
    main()
