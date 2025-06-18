import cv2
import numpy as np
import time
import argparse
from pathlib import Path  # Import the Path class
from concurrent.futures import ThreadPoolExecutor

# Assuming 'thinning' is a pre-existing optimized function
from linefiller.thinning import thinning
from kiseki.logging import logger, Profiler


class ColorizationPipeline:
    def __init__(self, min_merge_area=50, erosion_iterations=2):
        self.min_merge_area = min_merge_area
        self.erosion_iterations = erosion_iterations

    def read_image_to_binary(self, img_path: str) -> np.ndarray:
        print("[INFO] Reading image...")
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        self.original_bgr = (
            img[:, :, :3]
            if img.shape[2] >= 3
            else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        )

        if img.shape[2] == 4:
            source_channel = img[:, :, 3]
            _, binary_img = cv2.threshold(source_channel, 128, 255, cv2.THRESH_BINARY)
        else:
            source_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(
                source_channel, 220, 255, cv2.THRESH_BINARY_INV
            )

        # Convention: lines=0, fills=255.
        self.original_binary = cv2.bitwise_not(binary_img)
        return self.original_binary

    def watershed_segmentation(self, line_art_binary: np.ndarray) -> np.ndarray:
        print(
            f"[INFO] Performing Watershed... (Erosion: {self.erosion_iterations} iter)"
        )
        kernel = np.ones((3, 3), np.uint8)
        sure_fg = cv2.erode(line_art_binary, kernel, iterations=self.erosion_iterations)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1

        sure_bg = cv2.dilate(line_art_binary, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)
        markers[unknown == 255] = 0

        markers = cv2.watershed(self.original_bgr, markers)
        return markers.astype(np.int32)

    def heal_watershed_boundaries(self, watershed_map: np.ndarray) -> np.ndarray:
        print("[INFO] Healing watershed boundaries...")
        healed_map = watershed_map.copy()
        boundary_pixels = np.argwhere(healed_map == -1)
        for y, x in boundary_pixels:
            window = healed_map[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
            valid_labels = window[window > 0]
            if valid_labels.size > 0:
                u, c = np.unique(valid_labels, return_counts=True)
                healed_map[y, x] = u[np.argmax(c)]
            else:
                healed_map[y, x] = 0
        return healed_map

    def find_and_label_leftovers(
        self, primary_map: np.ndarray, original_binary: np.ndarray
    ) -> np.ndarray:
        print("[INFO] Rescue Pass: Finding leftover disconnected regions...")
        watershed_found_mask = (primary_map > 0).astype(np.uint8)
        leftovers_mask = cv2.subtract(original_binary, watershed_found_mask * 255)
        num_leftovers, leftover_labels = cv2.connectedComponents(leftovers_mask)
        if num_leftovers <= 1:
            return primary_map
        print(f"[INFO] Rescued {num_leftovers - 1} regions.")
        max_primary_label = np.max(primary_map)
        final_map = primary_map.copy()
        mask = leftover_labels > 0
        final_map[mask] = leftover_labels[mask] + max_primary_label
        return final_map

    def merge_small_regions(self, labels_map: np.ndarray) -> np.ndarray:
        print(
            f"[INFO] Post-Op Cleanup: Merging regions smaller than {self.min_merge_area} pixels..."
        )
        merged_map = labels_map.copy()
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            (labels_map > 0).astype(np.uint8)
        )
        small_region_labels = [
            i
            for i in range(1, num_labels)
            if stats[i, cv2.CC_STAT_AREA] < self.min_merge_area
        ]
        for label_id in small_region_labels:
            region_mask = (merged_map == label_id).astype(np.uint8)
            dilated_mask = cv2.dilate(region_mask, np.ones((3, 3), np.uint8))
            border_pixels = dilated_mask - region_mask
            neighbor_labels = merged_map[border_pixels == 1]
            neighbor_labels = neighbor_labels[neighbor_labels != 0]
            if neighbor_labels.size > 0:
                u, c = np.unique(neighbor_labels, return_counts=True)
                dominant_neighbor = u[np.argmax(c)]
                merged_map[merged_map == label_id] = dominant_neighbor
        return merged_map

    # --- NEW: Final, non-destructive step to guarantee line art fidelity ---
    def reimprint_line_art(self, labels_map: np.ndarray) -> np.ndarray:
        """
        Ensures the final data map perfectly preserves the original line art
        by stamping it on top of the final segmentation.
        """
        print(
            "[INFO] Finalizing map: Re-imprinting original line art for data integrity..."
        )
        final_map = labels_map.copy()
        # The original binary has lines as 0. This mask is our ground truth.
        line_mask = self.original_binary == 0
        final_map[line_mask] = 0
        return final_map

    def process(self, img_path: str) -> np.ndarray:
        """The complete hybrid pipeline, now with non-destructive output."""
        start_time = time.time()
        binary_art = self.read_image_to_binary(img_path)

        # Core segmentation pipeline
        watershed_map = self.watershed_segmentation(binary_art)
        healed_map = self.heal_watershed_boundaries(watershed_map)
        combined_map = self.find_and_label_leftovers(healed_map, binary_art)
        merged_map = self.merge_small_regions(combined_map)

        # Final non-destructive stamping of the ground truth
        final_map = self.reimprint_line_art(merged_map)

        end_time = time.time()
        print(f"[INFO] Total processing time: {end_time - start_time:.4f} seconds.")
        return final_map


def show_fill_map(fillmap: np.ndarray):
    max_label = np.max(fillmap)
    if max_label == 0:
        return np.zeros((*fillmap.shape, 3), dtype=np.uint8)
    colors = np.random.randint(0, 255, (max_label + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    return colors[fillmap]


def saveAll(fillmap: np.ndarray, PATH: str) -> None:
    """Save results with parallel processing."""
    # Use threading for parallel I/O
    with ThreadPoolExecutor(max_workers=2) as executor:
        # color+undertone
        f1 = show_fill_map(fillmap)
        future1 = executor.submit(cv2.imwrite, PATH / "fills_merged.png", f1)

        # undertone
        f2 = show_fill_map(thinning(fillmap))
        future2 = executor.submit(cv2.imwrite, PATH / "fills_merged_no_contour.png", f2)

        # Wait for completion
        future1.result()
        future2.result()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data-Preserving Line Art Colorization"
    )

    parser.add_argument(
        "-i", "--image", type=str, required=True, help="Path to input line art image."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",  # Default to a subdirectory
        help="Path to save the output colored map.",
    )
    parser.add_argument(
        "--merge_area",
        type=int,
        default=100,
        help="Max area for a region to be 'small' and merged.",
    )
    parser.add_argument(
        "--erosion",
        type=int,
        default=2,
        help="Erosion iterations for watershed markers (1=fine details, 3=more merging).",
    )

    args = parser.parse_args()
    # --- Pathlib Integration ---
    output_path = Path(args.output)
    # Create the parent directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    with Profiler("Trappedball Filling"):
        pipeline = ColorizationPipeline(
            min_merge_area=args.merge_area, erosion_iterations=args.erosion
        )
        fillmap = pipeline.process(img_path=args.image)

    saveAll(fillmap, output_path)
    print(f"[SUCCESS] Output saved to {args.output}")


if __name__ == "__main__":
    main()
