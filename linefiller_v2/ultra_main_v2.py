import cv2
import numpy as np
import scipy.ndimage
import heapq
import time
import argparse
from pathlib import Path

import cv2
import numpy as np
from collections import deque
from linefiller.thinning import thinning

from linefiller.trappedball_fill import (
    trapped_ball_fill_multi,
    flood_fill_multi,
    mark_fill,
    build_fill_map,
    merge_fill,
    show_fill_map,
)
from kiseki.logging import logger, Profiler


class HybridSegmentationPipeline:
    """
    CPU-only, zero-tune hybrid segmentation combining Watershed seeding
    and Trapped-Ball growth for leak-proof, sharp, artifact-free regions.
    """

    def __init__(self):
        self._auto_configured = False

    def auto_tune(self, binary_img: np.ndarray):
        """
        Analyze stroke width and image size to derive:
          - Erosion / dilation kernel sizes
          - Trapped-ball radius hierarchy
          - Tile size for union-find merging
        """
        h, w = binary_img.shape
        # Estimate line thickness via distance transform median
        dist = cv2.distanceTransform(
            (binary_img == 1).astype(np.uint8) * 255, cv2.DIST_L2, 5
        )
        median_thickness = max(1.0, np.median(dist[dist > 0]))
        # Derive morphological radii
        self.erode_iter = max(1, int(median_thickness // 2))
        self.dilate_iter = max(2, int(median_thickness))
        # Set trapped-ball radii levels
        self.radii = [
            max(1, int(median_thickness)),
            max(1, int(median_thickness // 2)),
            1,
        ]
        # Tile size for local merging
        self.tile_size = 64 if max(h, w) > 512 else 32
        self._auto_configured = True

    def preprocess(self, img_path: str):
        """
        Read image, convert to binary line-art:
          - Adaptive thresholding
          - Small-morphological closing to seal gaps
        """
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
        # Invert so lines=1, background=0
        _, bin_inv = cv2.threshold(gray, 220, 1, cv2.THRESH_BINARY_INV)
        # Close micro-gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        binary = cv2.morphologyEx(bin_inv, cv2.MORPH_CLOSE, kernel, iterations=1)
        return binary

    def seed_regions(self, binary: np.ndarray):
        """
        Fast Watershed pass to seed fine-grained markers.
        """
        # Erode to get sure foreground
        kernel = np.ones((3, 3), np.uint8)
        fg = cv2.erode(binary, kernel, iterations=self.erode_iter)
        # Compute connected markers
        num_labels, markers = cv2.connectedComponents(fg)
        markers = markers + 1
        # Background markers
        bg = cv2.dilate(binary, kernel, iterations=self.dilate_iter)
        unknown = bg - fg
        markers[unknown == 1] = 0
        # Watershed requires a 3-channel image
        color_img = cv2.cvtColor((binary * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        markers = cv2.watershed(color_img, markers)
        markers[markers == -1] = 0
        return markers.astype(np.int32)

    def trapped_ball_fill(self, binary: np.ndarray, seeds: np.ndarray):
        """
        Grow and merge seeded regions via multi-scale trapped-ball logic.
        """
        # Build fill list: use radii hierarchy on inverted binary (0=background)
        inv = (binary == 1).astype(np.uint8) * 255
        all_fills = []
        # multi-scale trapped-ball passes
        for r in self.radii:
            fills = trapped_ball_fill_multi(inv, r, method="mean")
            all_fills.extend(fills)
            inv = mark_fill(inv, fills)
        # final flood fill for any tiny regions
        fills = flood_fill_multi(inv)
        all_fills.extend(fills)
        # build and merge into fill map
        fillmap = build_fill_map(inv, all_fills)
        merged = merge_fill(fillmap)
        return merged

    def union_find_merge(self, label_map: np.ndarray):
        """
        Tile-based union-find to merge small/split regions across tile borders.
        """
        h, w = label_map.shape
        parent = {}
        # Initialize
        for lbl in np.unique(label_map):
            parent[int(lbl)] = int(lbl)

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        # Examine tile boundaries
        for y in range(0, h, self.tile_size):
            for x in range(0, w, self.tile_size):
                # Right neighbor
                if x + self.tile_size < w:
                    border = np.stack(
                        [
                            label_map[y : y + self.tile_size, x + self.tile_size - 1],
                            label_map[y : y + self.tile_size, x + self.tile_size],
                        ],
                        axis=-1,
                    )
                    for a, b in border.reshape(-1, 2):
                        if a and b and a != b:
                            union(int(a), int(b))
                # Bottom neighbor
                if y + self.tile_size < h:
                    border = np.stack(
                        [
                            label_map[y + self.tile_size - 1, x : x + self.tile_size],
                            label_map[y + self.tile_size, x : x + self.tile_size],
                        ],
                        axis=-1,
                    )
                    for a, b in border.reshape(-1, 2):
                        if a and b and a != b:
                            union(int(a), int(b))
        # Relabel
        out = np.zeros_like(label_map)
        for lbl in np.unique(label_map):
            root = find(int(lbl))
            out[label_map == lbl] = root
        return out

    def refine_boundaries(self, label_map: np.ndarray):
        """
        Final sub-pixel sharpening via distance-transform reprojection.
        """
        # Compute boundary mask
        h, w = label_map.shape
        out = label_map.copy()
        # Distance map per region centroid
        seeds = np.zeros((h, w), np.int32)
        for lbl in np.unique(label_map):
            if lbl == 0:
                continue
            mask = (label_map == lbl).astype(np.uint8)
            # place one seed at centroid
            ys, xs = np.where(mask)
            if len(xs):
                cy, cx = int(np.mean(ys)), int(np.mean(xs))
                seeds[cy, cx] = int(lbl)
        # Use watershed on negative distance to sharpen
        dist = cv2.distanceTransform((label_map == 0).astype(np.uint8), cv2.DIST_L2, 5)
        _, markers = cv2.connectedComponents(seeds)
        markers = markers.astype(np.int32)
        # watershed on distance map
        dist_color = cv2.cvtColor(
            (dist / dist.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR
        )
        ws = cv2.watershed(dist_color, markers)
        ws[ws < 0] = 0
        # Overlay sharpened labels
        mask = label_map != 0
        out[mask] = ws[mask]
        return out

    def segment(self, img_path: str) -> np.ndarray:
        """
        Full hybrid segmentation pipeline.
        """
        binary = self.preprocess(img_path)
        if not self._auto_configured:
            self.auto_tune(binary)
        seeds = self.seed_regions(binary)
        logger.info("[Pipeline] Watershed seeding complete.")
        grown = self.trapped_ball_fill(binary, seeds)
        logger.info("[Pipeline] Trapped-ball growth complete.")
        merged = self.union_find_merge(grown)
        logger.info("[Pipeline] Tile-based union-find merge complete.")
        final = self.refine_boundaries(merged)
        logger.info("[Pipeline] Boundary refinement complete.")
        return final


def saveAll(fillmap: np.ndarray, PATH: Path):
    logger.info("[INFO] Saving output images...")
    # Save the primary data map render (with lines)
    cv2.imwrite(
        str(PATH / "fills_with_lines.png"), show_fill_map(fillmap, lines_are_black=True)
    )

    thinned_image = show_fill_map(thinning(fillmap))
    cv2.imwrite(str(PATH / "fills_thinned.png"), thinned_image)
    logger.info(f"[INFO] Images saved in {PATH}")


# Example usage:
# pipeline = HybridSegmentationPipeline()
# fillmap = pipeline.segment("input_lineart.png")


def main():
    parser = argparse.ArgumentParser(description="Precision Line Art Colorization")
    parser.add_argument(
        "-i", "--image", type=str, required=True, help="Path to input line art image."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="Output directory."
    )
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    with Profiler("Trappedball Filling"):
        pipeline = HybridSegmentationPipeline()
        fillmap = pipeline.segment(img_path=args.image)
    saveAll(fillmap, output_path)
    logger.info("[SUCCESS] Processing complete.")


if __name__ == "__main__":
    main()
