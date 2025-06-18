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
from linefiller.thinning import thinning
from linefiller.trappedball_fill import (
    trapped_ball_fill_multi,
    flood_fill_multi,
    mark_fill,
    build_fill_map,
    merge_fill,
    show_fill_map,
)


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


def process_fast(image_path, output_path):
    """Fast processing pipeline that won't hang."""
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(exist_ok=True, parents=True)

    print(f"\nProcessing: {image_path}")
    start_total = time.perf_counter()

    # Load image
    image = read_line_2_np(image_path, channel=4)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("working.png", image)

    if image is None:
        print(f"Error: Cannot load {image_path}")
        return

    print(f"Image size: {image.shape[1]}x{image.shape[0]}")

    # Threshold
    start = time.perf_counter()
    _, binary = cv2.threshold(image, 220, 255, cv2.THRESH_BINARY)
    print(f"Threshold: {(time.perf_counter() - start)*1000:.1f} ms")

    # Multi-scale trapped-ball fill
    print("\nTrapped-ball fill:")
    result = binary.copy()

    for radius in [3, 2, 1]:
        start = time.perf_counter()
        filled = fast_trapped_ball_fill(result, radius)
        result = cv2.bitwise_and(result, filled)
        print(f"  Radius {radius}: {(time.perf_counter() - start)*1000:.1f} ms")

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
