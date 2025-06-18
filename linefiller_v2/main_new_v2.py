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
    def __init__(self, thinning_iterations=0, min_merge_area=50):
        self.thinning_iterations = thinning_iterations
        self.min_merge_area = min_merge_area

    # --- UPGRADE 1: Detail-preserving binarization ---
    def read_image_to_binary(self, img_path: str) -> np.ndarray:
        """
        Switched to adaptive thresholding to preserve fine lines and handle
        variable line weights, resulting in a much higher fidelity binary mask.
        """
        print("[INFO] Reading image with Adaptive Thresholding...")
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        if len(img.shape) > 2 and img.shape[2] == 4:
            # Use alpha channel as the primary source for line art
            source_channel = img[:, :, 3]
        else:
            source_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Adaptive thresholding is superior to global thresholding for line art.
        # It calculates different thresholds for different regions of the image.
        # ADAPTIVE_THRESH_GAUSSIAN_C is generally better for natural gradients.
        # Block size and C are key tuning parameters.
        binary_img = cv2.adaptiveThreshold(
            source_channel,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=21,  # Must be an odd number
            C=10,  # Constant subtracted from the mean
        )
        # Our convention is lines=0, fills=255. Inverting the mask achieves this.
        return cv2.bitwise_not(binary_img)

    def generate_fill_map(self, line_art: np.ndarray):
        print("[INFO] Generating base fill map with Connected Components...")
        num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
            line_art, 8, cv2.CV_32S
        )
        return num_labels, labels_map, stats

    def close_gaps(self, line_art: np.ndarray, radii: list) -> np.ndarray:
        print(f"[INFO] Closing gaps with radii: {radii}")
        inverted_art = cv2.bitwise_not(line_art)
        closed_inverted = inverted_art
        for r in radii:
            if r <= 0:
                continue
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)
            )
            # Using dst= avoids a copy, but let's be safe and work on a copy.
            closed_inverted = cv2.morphologyEx(
                closed_inverted, cv2.MORPH_CLOSE, kernel, iterations=1
            )
        return cv2.bitwise_not(closed_inverted)

    # --- UPGRADE 2: Smarter merging based on shared border length ---
    def merge_small_regions(
        self, num_labels: int, labels_map: np.ndarray, stats: np.ndarray
    ) -> np.ndarray:
        print(
            f"[INFO] Merging regions smaller than {self.min_merge_area} pixels with shared border logic..."
        )
        merged_map = labels_map.copy()

        # Pre-calculate a dilated version of the line mask for efficiency
        line_mask = (merged_map == 0).astype(np.uint8)
        dilated_line_mask = cv2.dilate(line_mask, np.ones((3, 3), np.uint8))

        small_region_labels = [
            i
            for i in range(1, num_labels)
            if stats[i, cv2.CC_STAT_AREA] < self.min_merge_area
        ]

        for label_id in small_region_labels:
            region_mask = (merged_map == label_id).astype(np.uint8)

            # Find neighbors by dilating the region and looking at what's underneath
            dilated_region = cv2.dilate(region_mask, np.ones((3, 3), np.uint8))
            border_pixels = dilated_region - region_mask

            # Get labels of all neighbors, including the line art (label 0)
            neighbor_labels = merged_map[border_pixels == 1]
            unique_neighbors = np.unique(neighbor_labels)

            # Filter out the line art itself from the list of candidates to merge with
            candidate_neighbors = unique_neighbors[unique_neighbors != 0]

            if candidate_neighbors.size > 0:
                best_neighbor = -1
                max_border_length = -1

                # Calculate shared border length for each neighbor
                for neighbor_id in candidate_neighbors:
                    neighbor_mask = (merged_map == neighbor_id).astype(np.uint8)
                    # The shared border is where the dilated region of this label
                    # intersects with the mask of the neighbor.
                    shared_border = cv2.bitwise_and(dilated_region, neighbor_mask)
                    border_length = np.sum(shared_border)

                    if border_length > max_border_length:
                        max_border_length = border_length
                        best_neighbor = neighbor_id

                if best_neighbor != -1:
                    merged_map[merged_map == label_id] = best_neighbor

        return merged_map

    # --- UPGRADE 3: A final refinement step for smoother edges ---
    def refine_edges(self, fill_map: np.ndarray) -> np.ndarray:
        """
        Smooths the boundaries between final colored regions to reduce blockiness
        from the merge operations.
        """
        print("[INFO] Refining final region boundaries...")
        refined_map = fill_map.copy()
        # Create a boundary mask: 1 pixel where regions meet, 0 otherwise
        dilated = cv2.dilate(fill_map.astype(np.float32), np.ones((3, 3), np.uint8))
        eroded = cv2.erode(fill_map.astype(np.float32), np.ones((3, 3), np.uint8))
        boundary_mask = (dilated - eroded).astype(bool)

        # For each boundary pixel, replace it with the mode of its neighbors
        # This is essentially a median filter on a label map.
        boundary_coords = np.argwhere(boundary_mask)
        for y, x in boundary_coords:
            # Create a 3x3 window of neighbors
            window = refined_map[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
            # Exclude the line art (0) from the mode calculation
            window_flat = window.flatten()
            window_flat = window_flat[window_flat != 0]
            if window_flat.size > 0:
                # Find the most frequent neighbor (mode)
                u, c = np.unique(window_flat, return_counts=True)
                refined_map[y, x] = u[np.argmax(c)]

        return refined_map

    def process(self, img_path: str, gap_closing_radii=[5, 2]) -> np.ndarray:
        start_time = time.time()
        binary_art = self.read_image_to_binary(img_path)
        closed_art = self.close_gaps(binary_art, gap_closing_radii)
        num_labels, labels_map, stats = self.generate_fill_map(closed_art)
        merged_map = self.merge_small_regions(num_labels, labels_map, stats)

        # Add the refinement step to the pipeline
        final_map = self.refine_edges(merged_map)

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
    parser = argparse.ArgumentParser(description="High-Fidelity Line Art Colorization")
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
        default=50,
        help="Max area for a region to be 'small' and merged.",
    )
    parser.add_argument(
        "--gaps",
        nargs="+",
        type=int,
        default=[3],
        help="List of gap radii to close. Start small.",
    )

    args = parser.parse_args()

    # --- Pathlib Integration ---
    output_path = Path(args.output)
    # Create the parent directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    with Profiler("Trappedball Filling"):
        pipeline = ColorizationPipeline(min_merge_area=args.merge_area)
        fillmap = pipeline.process(img_path=args.image, gap_closing_radii=args.gaps)
    saveAll(fillmap, output_path)

    print(f"[SUCCESS] Output saved to {output_path}")


if __name__ == "__main__":
    main()
