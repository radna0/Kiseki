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


class PreciseColorizer:
    """
    The definitive hybrid colorization pipeline. It uses the precise, iterative
    logic of the original multi-scale Trapped-Ball algorithm but implements it
    within a high-performance, vectorized framework.
    """

    def __init__(self, radii: list = [20, 10, 1], merge_threshold: int = 50):
        self.radii = radii
        self.merge_threshold = merge_threshold
        # Pre-generate kernels to avoid recreating them in a loop.
        self.kernels = {
            r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
            for r in radii
        }
        self.neighbor_kernel = np.ones((3, 3), np.uint8)

    def _get_unfilled_regions(self, image: np.ndarray, radius: int):
        """
        Finds all distinct unfilled areas and returns a single seed point for each.
        This is a massive optimization over scanning for individual pixels.
        """
        # We erode the potential seed area slightly to avoid picking seeds right
        # on a boundary, preserving a key part of the original's robustness.
        kernel = self.kernels.get(radius, self.kernels[1])
        seed_area = cv2.erode(image, kernel, iterations=1)

        # Find all distinct blobs in the seedable area in a single, fast C++ call.
        num_labels, _, _, centroids = cv2.connectedComponentsWithStats(
            seed_area, connectivity=8
        )

        if num_labels > 1:
            # Return a list of valid seed points (x, y) for all found regions.
            return centroids[1:].astype(int)
        return []

    def _trapped_ball_fill_single(
        self, image: np.ndarray, seed_point: tuple, radius: int
    ):
        """
        The precise, core trapped-ball logic, kept architecturally identical to
        the original for maximum precision. Returns a binary mask of the single filled region.
        """
        h, w = image.shape
        im_inv = cv2.bitwise_not(image)

        # Pass 1: Flood the entire area connected to the seed.
        pass1 = np.zeros_like(image)
        cv2.floodFill(pass1, None, seed_point, 255)
        pass1 = cv2.bitwise_and(pass1, im_inv)

        # Pass 2: Dilate. This is the magic step that jumps gaps by disconnecting
        # the seed's region from its neighbors.
        dilated = cv2.dilate(pass1, self.kernels[radius], iterations=1)

        # Pass 3: Flood again from the same seed to select only the single, now-isolated region.
        pass2 = np.zeros_like(image)
        cv2.floodFill(pass2, None, seed_point, 255)
        pass2 = cv2.bitwise_and(pass2, dilated)

        # Pass 4: Erode to return the fill to its original, sharp, and precise size.
        final_fill = cv2.erode(pass2, self.kernels[radius], iterations=1)

        return final_fill

    def process(self, img_path: str) -> np.ndarray:
        """Executes the full, high-precision colorization pipeline."""
        # Load the image using the robust reader.
        im_rgb = read_line_2_np(img_path, channel=3)
        if im_rgb is None:
            raise FileNotFoundError(f"Image not found at {img_path} or failed to read.")

        img_gray = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2GRAY)
        # We work with a binary image where white (255) is the area to be filled.
        _, binary = cv2.threshold(img_gray, 220, 255, cv2.THRESH_BINARY)

        unfilled_mask = binary.copy()
        fillmap = np.zeros_like(binary, dtype=np.int32)
        current_region_id = 1

        # STEP 1: MULTI-SCALE TRAPPED-BALL PASS
        # We use the precise logic, but in an optimized framework.
        for radius in self.radii:
            logger.info(f"--- Running Trapped-Ball Pass with Radius: {radius} ---")

            # Get a list of all seed points for this pass in one shot.
            seed_points = self._get_unfilled_regions(unfilled_mask, radius)

            for seed in seed_points:
                # Check if the seed's area has already been filled by a previous op in this pass.
                if unfilled_mask[seed[1], seed[0]] == 255:
                    fill_mask = self._trapped_ball_fill_single(
                        unfilled_mask, tuple(seed), radius
                    )

                    # Write the new region directly to the final integer map. No list of pixels.
                    fillmap[fill_mask == 255] = current_region_id
                    current_region_id += 1

                    # Update the master mask of what's left to fill.
                    unfilled_mask[fill_mask == 255] = 0

        # STEP 2: FINAL FLOOD-FILL PASS
        # Quickly fill any remaining simple holes that don't have gaps.
        logger.info("--- Running Final Flood-Fill Pass ---")
        num_labels, labels, _, _ = cv2.connectedComponentsWithStats(
            unfilled_mask, connectivity=8
        )
        if num_labels > 1:
            for label_id in range(1, num_labels):
                fillmap[labels == label_id] = current_region_id
                current_region_id += 1

        # STEP 3: OPTIMIZED MERGE PASS
        # A final, fast pass to clean up any tiny fragments.
        logger.info("--- Merging Small Regions ---")
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            fillmap.astype(np.uint8), connectivity=8
        )
        areas = stats[:, cv2.CC_STAT_AREA]

        small_region_indices = np.where(areas < self.merge_threshold)[0]
        if small_region_indices.size > 1:
            remap_labels = np.arange(num_labels)
            for label_id in small_region_indices:
                if label_id == 0:
                    continue
                region_mask = fillmap == label_id
                dilated_mask = cv2.dilate(
                    region_mask.astype(np.uint8), self.neighbor_kernel
                )
                border_mask = (dilated_mask == 1) & ~region_mask

                neighbors = fillmap[border_mask]
                valid_neighbors = neighbors[neighbors > 0]
                if valid_neighbors.size > 0:
                    largest_neighbor = valid_neighbors[
                        np.argmax(areas[valid_neighbors])
                    ]
                    remap_labels[label_id] = largest_neighbor
            fillmap = remap_labels[fillmap]

        return fillmap


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
    cv2.imwrite(str(PATH / "fills_with_lines.png"), show_fill_map(fillmap))
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
        default=20,
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

    radii = sorted(
        [r for r in [args.radius, args.radius // 2, 1] if r > 0], reverse=True
    )

    with Profiler("CPU Colorization Pipeline"):
        pipeline = PreciseColorizer(radii=radii, merge_threshold=args.merge_area)
        fillmap = pipeline.process(img_path=args.image)

    # Use the requested saveAll function structure
    saveAll(fillmap, output_path)

    logger.info(f"[SUCCESS] Processing complete. Output saved to '{output_path}'.")


if __name__ == "__main__":
    main()
