#!/usr/bin/env python3
"""
High-Speed, CPU-Optimized Line Art Segmentation Engine

This script provides a definitive, high-performance pipeline for segmenting 2D
line art to detect all colorable regions. It replaces slow, iterative
flood-fill methods with a modern, bulk-processing approach for maximum speed
and reliability.

The core pipeline is:
1.  **Binarize:** Load the image and create a clean, binary line art mask.
2.  **Close Gaps:** Use multi-scale morphological closing to seal gaps in the
    line art without damaging details. This replaces "trapped-ball fill."
3.  **Segment All:** A SINGLE call to `cv2.connectedComponentsWithStats` finds
    every single colorable region and calculates its properties instantly.
4.  **Refine:** A fast-merge function cleans up tiny, irrelevant noise regions,
    and a bounded thinning function sharpens edges.
"""
import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from PIL import Image


def load_and_binarize(image_path: Path):
    """
    Loads an image and creates a clean binary mask.
    Returns:
        binary_lines: Image where lines are 0 (black), background is 255.
        binary_fillable: The inverse, where fillable areas are 255 (white).
    """
    try:
        img = Image.open(image_path)
    except FileNotFoundError:
        print(f"FATAL: Image not found at {image_path}")
        return None, None

    # Ensure image is in a standard format (grayscale) for reliable processing
    if img.mode == "RGBA":
        # If alpha exists, it's the most reliable source for lines
        alpha = np.array(img.split()[-1])
        mask = alpha > 100
    else:
        # For RGB or other modes, convert to grayscale and threshold
        grayscale_img = img.convert("L")
        mask = np.array(grayscale_img) < 150  # Dark pixels are lines

    binary_lines = np.where(mask, 0, 255).astype(np.uint8)
    binary_fillable = ~binary_lines
    return binary_lines, binary_fillable


def close_gaps(binary_fillable, radii=[15, 7, 3]):
    """
    Seals gaps in the line art using sequential morphological closing.
    Larger radii close bigger gaps, smaller radii refine details.
    This is the fast, non-iterative replacement for "trapped-ball".
    """
    closed = binary_fillable
    for r in radii:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel)
    return closed


def merge_small_regions(labels, stats, min_area=100):
    """
    Merges noisy regions smaller than `min_area` into their largest neighbor.
    Uses the pre-calculated `stats` for extreme efficiency.
    """
    # Create a map of {label_id: area} for fast lookups.
    # Label 0 is the background (lines), so we ignore it.
    area_map = {i: stats[i, cv2.CC_STAT_AREA] for i in range(1, stats.shape[0])}

    # Find all regions to be merged
    regions_to_merge = [i for i, area in area_map.items() if area < min_area]
    if not regions_to_merge:
        return labels

    print(f"   - Found {len(regions_to_merge)} small noise regions to merge.")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    for region_id in regions_to_merge:
        # Find neighbors using a fast dilation
        mask = (labels == region_id).astype(np.uint8)
        dilated_mask = cv2.dilate(mask, kernel)
        neighbor_ids = np.unique(labels[dilated_mask != mask])
        # Filter out the background (0) and the region itself
        valid_neighbors = [nid for nid in neighbor_ids if nid != 0 and nid != region_id]

        if not valid_neighbors:
            continue

        # Find the largest neighbor to merge into
        largest_neighbor = max(valid_neighbors, key=lambda nid: area_map.get(nid, 0))
        labels[labels == region_id] = largest_neighbor

    return labels


def fast_thinning(fill_map, binary_lines, max_iterations=3):
    """
    Safely bleeds colors into line art pixels with a hard iteration limit
    and early-exit logic to guarantee performance.
    """
    result = fill_map.astype(np.uint16)  # Use 16-bit to support >255 regions
    uncolored_lines = (binary_lines == 0) & (result == 0)

    if not np.any(uncolored_lines):
        return result

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for i in range(max_iterations):
        last_uncolored_count = np.count_nonzero(result == 0)

        influence_map = cv2.dilate(result, kernel)
        result[uncolored_lines] = influence_map[uncolored_lines]

        # Early exit if the image has stabilized
        if np.count_nonzero(result == 0) == last_uncolored_count:
            print(f"   - Thinning converged in {i+1} iterations.")
            break

    return result


def main():
    parser = argparse.ArgumentParser(
        description="A blazingly fast CPU-based line art segmenter.",
        epilog="Example: python segmenter.py my_drawing.png -o output",
    )
    parser.add_argument(
        "input_path", type=Path, help="Path to the input line art image."
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=Path,
        default=Path("output"),
        help="Directory to save results.",
    )
    args = parser.parse_args()

    if not args.input_path.exists():
        print(f"FATAL: Input file not found at '{args.input_path}'")
        return

    print("🚀 Starting High-Speed Segmentation Pipeline...")
    total_start_time = time.perf_counter()

    # 1. Load Image and Binarize
    binary_lines, binary_fillable = load_and_binarize(args.input_path)
    if binary_lines is None:
        return

    # 2. Close Gaps in Line Art
    closed_image = close_gaps(binary_fillable)

    # 3. Segment ALL Regions at Once
    # This is the core of the algorithm. One call does what used to take thousands of loops.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        closed_image, connectivity=4
    )
    print(f"✅ Found {num_labels - 1} initial color regions.")

    # 4. Refine the Segmentation
    merged_labels = merge_small_regions(labels, stats, min_area=100)
    final_map = fast_thinning(merged_labels, binary_lines)

    # 5. Save Results
    args.output_path.mkdir(exist_ok=True, parents=True)

    # Save the raw data map (useful for other programs)
    # Use 16-bit PNG to support potentially thousands of regions
    map_save_path = args.output_path / f"{args.input_path.stem}_map.png"
    cv2.imwrite(str(map_save_path), final_map.astype(np.uint16))

    # Save a colored visualization for easy checking
    vis_save_path = args.output_path / f"{args.input_path.stem}_visualization.png"
    colors = np.random.randint(50, 255, size=(final_map.max() + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]  # Ensure lines (region 0) are black
    visualization = colors[final_map]
    cv2.imwrite(str(vis_save_path), visualization)

    total_time_ms = (time.perf_counter() - total_start_time) * 1000
    print(f"🏁 Pipeline finished in {total_time_ms:.2f} ms.")
    print(f"   - Final map saved to: {map_save_path}")
    print(f"   - Visualization saved to: {vis_save_path}")


if __name__ == "__main__":
    main()
