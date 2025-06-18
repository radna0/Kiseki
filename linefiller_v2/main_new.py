import cv2
import numpy as np
import time
import argparse
from linefiller.thinning import thinning


class ColorizationPipeline:
    def __init__(self, thinning_iterations=3, min_merge_area=50):
        self.thinning_iterations = thinning_iterations
        self.min_merge_area = min_merge_area

    def read_image_to_binary(self, img_path: str) -> np.ndarray:
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        if img.shape[2] == 4:
            alpha = img[:, :, 3]
            # We want lines to be 0 and fills to be 255.
            # threshold gives us lines=255, so we invert it.
            _, binary_mask = cv2.threshold(alpha, 100, 255, cv2.THRESH_BINARY)
            binary_img = cv2.bitwise_not(binary_mask)
        else:
            grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # THRESH_BINARY_INV gives us lines=255, fills=0. We want the opposite.
            _, binary_img = cv2.threshold(grayscale, 220, 255, cv2.THRESH_BINARY)

        return binary_img

    # --- CHANGE 1: Corrected logic and new return signature ---
    def generate_fill_map(self, line_art: np.ndarray):
        """
        This is the core of the new pipeline.
        The input `line_art` is already what we need: lines are 0 (background for the
        algorithm) and fills are 255 (the components to be labeled).
        """
        print("[INFO] Generating base fill map with Connected Components...")
        # `line_art` has fills=255, lines=0. This is the correct input.
        # We now return the stats array so it can be used later without recalculation.
        num_labels, labels_map, stats, _ = cv2.connectedComponentsWithStats(
            line_art, 8, cv2.CV_32S
        )
        return num_labels, labels_map, stats

    def close_gaps(self, line_art: np.ndarray, radii: list) -> np.ndarray:
        print(f"[INFO] Closing gaps with radii: {radii}")
        # The input here has lines=0, fills=255. Closing works by filling holes (255)
        # surrounded by the foreground (0), which is not what we want. We must
        # operate on the inverted image.
        inverted_art = cv2.bitwise_not(line_art)
        closed_inverted = inverted_art.copy()
        for r in radii:
            if r <= 0:
                continue
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)
            )
            cv2.morphologyEx(
                closed_inverted,
                cv2.MORPH_CLOSE,
                kernel,
                dst=closed_inverted,
                iterations=1,
            )
        # Invert back to the lines=0, fills=255 convention.
        return cv2.bitwise_not(closed_inverted)

    # --- CHANGE 3: Simplified logic, no more recalculation ---
    def merge_small_regions(
        self, num_labels: int, labels_map: np.ndarray, stats: np.ndarray
    ) -> np.ndarray:
        """
        Intelligently merges small, insignificant regions into their neighbors.
        It now receives the stats array directly, ensuring label consistency.
        """
        print(f"[INFO] Merging regions smaller than {self.min_merge_area} pixels...")
        merged_map = labels_map.copy()

        # `stats` comes from the same operation that created `labels_map`. The IDs are consistent.
        # Label 0 is the background (the lines). We iterate from 1.
        for label_id in range(1, num_labels):
            if stats[label_id, cv2.CC_STAT_AREA] < self.min_merge_area:
                region_mask = (merged_map == label_id).astype(np.uint8)
                kernel = np.ones((3, 3), np.uint8)
                dilated_mask = cv2.dilate(region_mask, kernel, iterations=2)
                border_pixels = dilated_mask - region_mask

                # Get labels of neighbors
                neighbor_labels = merged_map[border_pixels == 1]
                # Filter out the line art label (0)
                neighbor_labels = neighbor_labels[neighbor_labels != 0]

                if neighbor_labels.size > 0:
                    unique_neighbors, counts = np.unique(
                        neighbor_labels, return_counts=True
                    )
                    dominant_neighbor = unique_neighbors[np.argmax(counts)]
                    merged_map[merged_map == label_id] = dominant_neighbor

        return merged_map

    # ... (thin_lines and other methods remain the same) ...
    def thin_lines(self, fill_map: np.ndarray) -> np.ndarray:
        if self.thinning_iterations <= 0:
            return fill_map
        print(f"[INFO] Thinning lines with {self.thinning_iterations} iterations...")
        line_mask = (fill_map == 0).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        thinned_lines = cv2.erode(
            line_mask, kernel, iterations=self.thinning_iterations
        )
        final_map = fill_map.copy()
        final_map[thinned_lines == 0] = fill_map.max() + 1
        final_map[line_mask != 0] = 0
        final_map[thinned_lines != 0] = fill_map[thinned_lines != 0]
        return final_map

    # --- CHANGE 2: Updated pipeline data flow ---
    def process(self, img_path: str, gap_closing_radii=[5, 2]) -> np.ndarray:
        """
        The main pipeline execution flow.
        """
        start_time = time.time()

        # 1. Read and Binarize (convention: lines=0, fills=255)
        binary_art = self.read_image_to_binary(img_path)

        # 2. Close Gaps
        closed_art = self.close_gaps(binary_art, gap_closing_radii)

        # 3. Generate Fill Map and Stats in one shot.
        num_labels, labels_map, stats = self.generate_fill_map(closed_art)

        # 4. Merge Small Regions using the data from the previous step.
        final_map = self.merge_small_regions(num_labels, labels_map, stats)

        # 5. Thinning (Optional visual step)
        final_map = thinning(final_map, max_iter=5)

        end_time = time.time()
        print(f"[INFO] Total processing time: {end_time - start_time:.4f} seconds.")
        return final_map


# ... (The rest of the file, show_fill_map, main, etc., remains the same) ...
def show_fill_map(fillmap: np.ndarray):
    """Marks filled areas with random colors for visualization."""
    max_label = np.max(fillmap)
    if max_label == 0:
        return np.zeros((*fillmap.shape, 3), dtype=np.uint8)
    np.random.seed(0)
    colors = np.random.randint(0, 255, (max_label + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    return colors[fillmap]


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-architected Line Art Colorization")
    parser.add_argument(
        "-i", "--image", type=str, required=True, help="Path to input line art image."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.png",
        help="Path to save the output colored map.",
    )
    parser.add_argument(
        "--thinning",
        type=int,
        default=100,
        help="Number of thinning iterations for the lines (0-5).",
    )
    parser.add_argument(
        "--merge_area",
        type=int,
        default=50,
        help="Max area for a region to be considered 'small' and merged.",
    )
    parser.add_argument(
        "--gaps",
        nargs="+",
        type=int,
        default=[5, 2],
        help="List of gap radii to close.",
    )

    args = parser.parse_args()

    pipeline = ColorizationPipeline(
        thinning_iterations=args.thinning, min_merge_area=args.merge_area
    )

    fillmap = pipeline.process(img_path=args.image, gap_closing_radii=args.gaps)

    output_image = show_fill_map(fillmap)
    cv2.imwrite(args.output, output_image)
    print(f"[SUCCESS] Output saved to {args.output}")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
