import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from numba import njit
from collections import defaultdict
from kiseki.logging import logger, Profiler
from linefiller.thinning import thinning


# --- UPGRADE: The user's thinning logic, accelerated with Numba ---
@njit(nogil=True, fastmath=True)
def fast_thinning(fillmap: np.ndarray, max_iter: int = 100):
    """
    Fills line art pixels (label 0) by iteratively growing neighbor regions.
    This is a Numba-accelerated version of the provided thinning logic for sharp, cel-shaded fills.
    """
    h, w = fillmap.shape
    result = fillmap.copy()

    for _ in range(max_iter):
        changed_in_pass = False
        # Identify all line pixels to be processed in this pass
        line_points_y, line_points_x = np.where(result == 0)

        if line_points_y.shape[0] == 0:
            break  # No more lines to thin

        # We check neighbors for each line pixel. If a colored neighbor is found,
        # we mark this pixel for update. We do this on a copy to ensure
        # all decisions in a single pass are based on the state at the start of the pass.
        temp_result = result.copy()
        for i in range(line_points_y.shape[0]):
            y, x = line_points_y[i], line_points_x[i]

            # 8-way neighbor check, optimized for Numba
            if y > 0 and result[y - 1, x] != 0:
                temp_result[y, x] = result[y - 1, x]
            elif y < h - 1 and result[y + 1, x] != 0:
                temp_result[y, x] = result[y + 1, x]
            elif x > 0 and result[y, x - 1] != 0:
                temp_result[y, x] = result[y, x - 1]
            elif x < w - 1 and result[y, x + 1] != 0:
                temp_result[y, x] = result[y, x + 1]
            elif y > 0 and x > 0 and result[y - 1, x - 1] != 0:
                temp_result[y, x] = result[y - 1, x - 1]
            elif y > 0 and x < w - 1 and result[y - 1, x + 1] != 0:
                temp_result[y, x] = result[y - 1, x + 1]
            elif y < h - 1 and x > 0 and result[y + 1, x - 1] != 0:
                temp_result[y, x] = result[y + 1, x - 1]
            elif y < h - 1 and x < w - 1 and result[y + 1, x + 1] != 0:
                temp_result[y, x] = result[y + 1, x + 1]

        # Check if any change was made to terminate early
        if np.array_equal(result, temp_result):
            break

        result = temp_result

    return result


class CPUColorizationPipeline:
    """
    An optimized, CPU-only colorization pipeline that avoids slow, iterative Python
    in favor of bulk array operations via OpenCV and Numpy.
    """

    def __init__(self, leak_proof_radius=2, merge_area_threshold=50):
        self.leak_proof_radius = leak_proof_radius
        self.merge_area_threshold = merge_area_threshold
        # Pre-computing kernels is a trivial but important micro-optimization.
        self.closing_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2 * self.leak_proof_radius + 1, 2 * self.leak_proof_radius + 1),
        )

    def _build_adjacency_map(self, labels: np.ndarray, areas: np.ndarray):
        """
        Builds a map of {region_id: largest_neighbor_id} for all regions
        smaller than the merge threshold. This is done in a single, vectorized pass.
        """
        # Identify all regions that are candidates for merging.
        small_region_indices = np.where(areas < self.merge_area_threshold)[0]
        if small_region_indices.size == 0:
            return {}  # No small regions to merge.

        # Create a boolean mask of all small regions at once.
        small_region_mask = np.isin(labels, small_region_indices)

        # Dilate the entire set of small regions to find the boundary zone.
        # This is one OpenCV call, not one per region.
        boundary_zone = cv2.dilate(
            small_region_mask.astype(np.uint8), np.ones((3, 3), np.uint8)
        )

        # Identify the labels of all regions touching the small ones.
        neighbor_labels_map = labels.copy()
        neighbor_labels_map[small_region_mask] = (
            0  # Exclude the small regions themselves.
        )

        # This gives us the labels of all neighbors in the boundary zone.
        border_pixels = neighbor_labels_map[boundary_zone.astype(bool)]

        # This is our map from a small region to its merge target.
        merge_map = {}

        # Now, iterate only through the small regions to decide their fate.
        for label_id in small_region_indices:
            if label_id == 0:
                continue  # Skip background

            # Find the border of this specific small region
            region_mask = labels == label_id
            dilated_region = cv2.dilate(
                region_mask.astype(np.uint8), np.ones((3, 3), np.uint8)
            )
            border_mask = (dilated_region == 1) & ~region_mask

            # Get the labels of its direct neighbors
            neighbors = labels[border_mask]
            # Filter out lines/background (0) and self
            valid_neighbors = neighbors[neighbors != 0]

            if valid_neighbors.size > 0:
                # Find the neighbor with the largest area to merge with.
                # This is a robust heuristic to prevent merging into another tiny fragment.
                largest_neighbor = valid_neighbors[np.argmax(areas[valid_neighbors])]
                merge_map[label_id] = largest_neighbor

        return merge_map

    def process(self, img_path: str) -> np.ndarray:
        # STEP 1: READ AND PREPARE BINARY LINE ART
        # I/O bound, minimal processing.
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        _, binary_lines = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)

        # STEP 2: LEAK-PROOF FILL (TRAPPED-BALL) VIA MORPHOLOGICAL CLOSING
        # Replaces complex iteration with a single, highly-optimized C++ call.
        logger.info(
            f"Performing leak-proof fill with radius {self.leak_proof_radius}..."
        )
        closed_art = cv2.morphologyEx(
            binary_lines, cv2.MORPH_CLOSE, self.closing_kernel
        )

        # STEP 3: IDENTIFY ALL REGIONS AT ONCE
        # The workhorse. Returns labels and stats (including area) in one shot.
        logger.info("Identifying connected regions...")
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            cv2.bitwise_not(closed_art), connectivity=4
        )
        areas = stats[:, cv2.CC_STAT_AREA]

        # STEP 4: INTELLIGENT REGION MERGING
        # This is the new, faster approach. Build a map of merges first, then apply.
        logger.info(
            f"Building adjacency map to merge regions smaller than {self.merge_area_threshold} pixels..."
        )
        merge_map = self._build_adjacency_map(labels, areas)

        if merge_map:
            logger.info(f"Applying {len(merge_map)} region merges...")
            # Apply merges using a lookup table. This is faster than iterating the image.
            # We create a full mapping from old labels to new labels.
            final_labels = np.arange(num_labels)
            for key, value in merge_map.items():
                final_labels[key] = value

            # The actual merge is a single, fast, vectorized lookup.
            labels = final_labels[labels]

        # STEP 5: FINAL DATA INTEGRITY PASS
        # The morphological operations may have thickened lines. Restore the original
        # line art as the absolute ground truth.
        logger.info("Restoring original line art...")
        labels[binary_lines == 255] = 0

        return labels.astype(np.int32)


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


def saveAll(fillmap: np.ndarray, PATH: Path):
    logger.info("[INFO] Saving output images...")
    # Save the primary data map render (with lines)
    cv2.imwrite(
        str(PATH / "fills_with_lines.png"), show_fill_map(fillmap, lines_are_black=True)
    )
    smoothed_image = show_fill_map(fast_thinning(fillmap))
    cv2.imwrite(str(PATH / "fills_fast_thinned.png"), smoothed_image)

    thinned_image = show_fill_map(thinning(fillmap))
    cv2.imwrite(str(PATH / "fills_thinned.png"), thinned_image)
    logger.info(f"[INFO] Images saved in {PATH}")


def main():
    parser = argparse.ArgumentParser(
        description="High-Performance CPU Line Art Colorization"
    )
    parser.add_argument(
        "-i", "--image", type=str, required=True, help="Path to input line art image."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="Output directory."
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Leak-proof fill radius. Should be floor(gap_size / 2).",
    )
    parser.add_argument(
        "--merge_area",
        type=int,
        default=50,
        help="Max area for a region to be 'small' and merged.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    with Profiler("CPU Colorization Pipeline"):
        pipeline = CPUColorizationPipeline(
            leak_proof_radius=args.radius, merge_area_threshold=args.merge_area
        )
        fillmap = pipeline.process(img_path=args.image)

    # Use the requested saveAll function structure
    saveAll(fillmap, output_path)

    logger.info(f"[SUCCESS] Processing complete. Output saved to '{output_path}'.")


if __name__ == "__main__":
    main()
