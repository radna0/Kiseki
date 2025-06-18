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
    def __init__(self, min_merge_area=50):
        # We no longer need thinning iterations in the main class
        self.min_merge_area = min_merge_area

    def read_image_to_binary(self, img_path: str) -> np.ndarray:
        print("[INFO] Reading image...")
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")

        # Keep original image for watershed input
        self.original_image_bgr = (
            img[:, :, :3]
            if img.shape[2] >= 3
            else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        )

        if img.shape[2] == 4:
            source_channel = img[:, :, 3]
            _, binary_img = cv2.threshold(source_channel, 100, 255, cv2.THRESH_BINARY)
        else:
            source_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(
                source_channel, 220, 255, cv2.THRESH_BINARY_INV
            )

        # Convention: lines=0, fills=255. `binary_img` from threshold is lines=255, fills=0
        return cv2.bitwise_not(binary_img)

    def watershed_segmentation(self, line_art_binary: np.ndarray) -> np.ndarray:
        """
        This is the new core of the pipeline. It replaces gap closing,
        component analysis, and merging with a single, powerful algorithm.
        """
        print("[INFO] Performing Marker-Based Watershed Segmentation...")

        # 1. Identify the definite foreground (unambiguous fill areas)
        # We erode the fill mask to get the "cores" of the regions.
        # This ensures our markers are far from any contested boundaries.
        kernel = np.ones((3, 3), np.uint8)
        sure_fg = cv2.erode(line_art_binary, kernel, iterations=3)

        # 2. Generate the markers for the watershed algorithm
        # `connectedComponents` will give us a map of these "sure" regions.
        ret, markers = cv2.connectedComponents(sure_fg)

        # The watershed algorithm uses 0 for boundaries. We add 1 to all labels
        # so that our `sure_fg` markers start at 1.
        markers = markers + 1

        # 3. Identify the definite background (the line art itself).
        # We dilate the lines to make them thicker, creating a definite "unknown" zone
        # between the lines and the foreground markers.
        sure_bg = cv2.dilate(line_art_binary, kernel, iterations=3)
        unknown = cv2.subtract(sure_bg, sure_fg)

        # Mark the unknown region as 0. This is what watershed will fill in.
        markers[unknown == 255] = 0

        # 4. Run the watershed algorithm.
        # It needs a 3-channel image to operate on.
        # It will modify the `markers` array in-place.
        markers = cv2.watershed(self.original_image_bgr, markers)

        # The output `markers` map now contains the final segmentation.
        # The boundaries are marked with -1. We'll set them to 0 (our line convention).
        final_map = markers.copy()
        final_map[final_map == -1] = 0

        # The rest of the labels are off by one due to the watershed prep. Let's remap them.
        unique_labels = np.unique(final_map)
        # Create a mapping from old labels to a new compact set of labels (0, 1, 2, ...)
        remap = {
            old_label: new_label for new_label, old_label in enumerate(unique_labels)
        }
        # Apply the mapping
        for old, new in remap.items():
            final_map[markers == old] = new

        return final_map.astype(np.int32)

    def process(self, img_path: str) -> np.ndarray:
        """
        The main pipeline, now centered around the watershed algorithm.
        """
        start_time = time.time()

        # 1. Read and Binarize
        binary_art = self.read_image_to_binary(img_path)

        # 2. Segment using Watershed
        # This single step replaces close_gaps, generate_fill_map, and merge_small_regions
        final_map = self.watershed_segmentation(binary_art)

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
        description="High-Fidelity Watershed-Based Line Art Colorization"
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

    args = parser.parse_args()

    # --- Pathlib Integration ---
    output_path = Path(args.output)
    # Create the parent directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    with Profiler("Trappedball Filling"):
        pipeline = ColorizationPipeline()
        fillmap = pipeline.process(img_path=args.image)
    saveAll(fillmap, output_path)

    logger.info(f"[SUCCESS] Output saved to {output_path}")


if __name__ == "__main__":
    main()
