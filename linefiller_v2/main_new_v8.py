import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import defaultdict
from linefiller.thinning import thinning


# --- UPGRADE: A proper, high-quality thinning/smoothing function ---
def smooth_fill(fillmap_with_lines: np.ndarray) -> np.ndarray:
    """
    Creates a smooth, contourless fill by in-painting the line art.
    This uses a sophisticated algorithm for high-quality results.
    """
    print("[INFO] Performing high-quality smoothing (in-painting)...")

    # Get a color representation first
    colored_view = show_fill_map(fillmap_with_lines, lines_are_black=False)

    # Create a mask of the lines (where label is 0)
    line_mask = (fillmap_with_lines == 0).astype(np.uint8)

    # Use Navier-Stokes based In-painting for the highest quality result
    # It intelligently propagates color from the edges of the mask.
    smoothed_image = cv2.inpaint(colored_view, line_mask, 3, cv2.INPAINT_NS)

    return smoothed_image


class ColorizationPipeline:
    def __init__(self, min_merge_area=200, erosion_iterations=2):
        self.min_merge_area = min_merge_area
        self.erosion_iterations = erosion_iterations

    def read_image_to_binary(self, img_path: str):
        # This function is sound, no changes needed.
        print("[INFO] Reading image...")
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
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
        self.original_binary = cv2.bitwise_not(binary_img)
        return self.original_binary

    # --- UPGRADE: Replaced healing and rescue with a single, superior in-painting step ---
    def inpaint_and_resolve_boundaries(self, watershed_map: np.ndarray) -> np.ndarray:
        """
        Uses a distance transform and a second watershed pass to perfectly fill
        the boundaries and any leftover regions without spilling.
        """
        print("[INFO] Resolving boundaries with Distance Transform + Watershed...")

        markers = watershed_map.copy()
        markers[markers == -1] = 0
        to_fill_mask = (self.original_binary == 0) | (markers == 0)

        dist_transform = cv2.distanceTransform(
            to_fill_mask.astype(np.uint8), cv2.DIST_L2, 5
        )

        # --- FIX IS HERE ---
        # The watershed 'src' image MUST be 8-bit, 3-channel (CV_8UC3).
        # We must normalize the float distance transform to the 0-255 range
        # and convert it properly.
        cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)
        dist_8u = dist_transform.astype(np.uint8)
        dist_bgr = cv2.cvtColor(dist_8u, cv2.COLOR_GRAY2BGR)

        # Now we pass the correctly formatted image to the watershed function.
        cv2.watershed(dist_bgr, markers)

        markers[markers == -1] = 0
        return markers.astype(np.int32)

    def watershed_segmentation(self, line_art_binary: np.ndarray) -> np.ndarray:
        """
        Performs the primary segmentation using the watershed algorithm.
        This is the engine of the pipeline.
        """
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

    def build_region_adjacency_graph(self, labels_map: np.ndarray):
        # This function is sound, no changes needed.
        print("[INFO] Building Region Adjacency Graph...")
        graph = defaultdict(int)
        h, w = labels_map.shape
        for y in range(h):
            for x in range(w - 1):
                p1, p2 = labels_map[y, x], labels_map[y, x + 1]
                if p1 != p2 and p1 != 0 and p2 != 0:
                    graph[tuple(sorted((p1, p2)))] += 1
        for y in range(h - 1):
            for x in range(w):
                p1, p2 = labels_map[y, x], labels_map[y + 1, x]
                if p1 != p2 and p1 != 0 and p2 != 0:
                    graph[tuple(sorted((p1, p2)))] += 1
        return graph

    def merge_regions_iteratively(self, labels_map: np.ndarray, graph: dict):
        """A truly iterative and stable merge process using a Union-Find-like mapping."""
        print(
            f"[INFO] Iteratively merging regions smaller than {self.min_merge_area} pixels..."
        )

        # Get initial stats
        unique_labels, counts = np.unique(labels_map, return_counts=True)
        areas = dict(zip(unique_labels, counts))
        # Parent pointer for each label. Initially, each label is its own parent.
        parent = {label: label for label in unique_labels if label > 0}

        while True:
            merges_made_this_pass = 0

            # Always process smallest regions first
            # We operate on the current state of areas
            sorted_labels = sorted(
                [l for l in parent if parent[l] == l and l in areas],
                key=lambda l: areas[l],
            )

            for label_id in sorted_labels:
                if areas[label_id] < self.min_merge_area:
                    best_neighbor = -1
                    max_border = -1
                    # Find best neighbor from the static graph
                    for edge, border_length in graph.items():
                        if label_id in edge:
                            neighbor_id = edge[0] if edge[1] == label_id else edge[1]
                            # Find the ultimate parent of the neighbor
                            root_neighbor = neighbor_id
                            while parent[root_neighbor] != root_neighbor:
                                root_neighbor = parent[root_neighbor]

                            if label_id != root_neighbor and border_length > max_border:
                                max_border = border_length
                                best_neighbor = root_neighbor

                    if best_neighbor != -1:
                        # Perform the merge: point this label's parent to the neighbor
                        # and transfer its area.
                        parent[label_id] = best_neighbor
                        areas[best_neighbor] += areas[label_id]
                        areas[label_id] = 0
                        merges_made_this_pass += 1

            if merges_made_this_pass == 0:
                print("[INFO] Merge process has converged.")
                break

        print("[INFO] Applying final merge mapping...")
        # Create final map by resolving all parent pointers
        final_map = labels_map.copy()
        for label_id in range(1, len(parent) + 1):
            if label_id in parent:
                root = label_id
                while parent[root] != root:
                    root = parent[root]
                final_map[labels_map == label_id] = root

        return final_map

    def process(self, img_path: str) -> np.ndarray:
        start_time = time.time()
        binary_art = self.read_image_to_binary(img_path)

        # 1. Core segmentation
        watershed_map = self.watershed_segmentation(binary_art)

        # 2. High-quality boundary and leftover resolution
        resolved_map = self.inpaint_and_resolve_boundaries(watershed_map)

        # 3. Intelligent merging based on a graph
        region_graph = self.build_region_adjacency_graph(resolved_map)
        merged_map = self.merge_regions_iteratively(resolved_map, region_graph)

        # 4. Final data integrity pass
        final_map = merged_map.copy()
        final_map[self.original_binary == 0] = 0

        end_time = time.time()
        print(f"[INFO] Total processing time: {end_time - start_time:.4f} seconds.")
        return final_map


def show_fill_map(fillmap: np.ndarray, lines_are_black=True):
    # This render function now supports not having black lines
    max_label = np.max(fillmap)
    if max_label == 0:
        return np.zeros((*fillmap.shape, 3), dtype=np.uint8)
    np.random.seed(0)
    colors = np.random.randint(50, 255, (int(max_label) + 1, 3), dtype=np.uint8)
    if lines_are_black:
        colors[0] = [0, 0, 0]
    return colors[fillmap.astype(int)]


def saveAll(fillmap: np.ndarray, PATH: Path):
    print("[INFO] Saving output images...")
    # Save the primary data map render (with lines)
    cv2.imwrite(
        str(PATH / "fills_with_lines.png"), show_fill_map(fillmap, lines_are_black=True)
    )
    # Create and save the "smooth" version using high-quality in-painting
    smoothed_image = smooth_fill(fillmap)
    cv2.imwrite(str(PATH / "fills_smooth.png"), smoothed_image)

    thinned_image = show_fill_map(thinning(fillmap))
    cv2.imwrite(str(PATH / "fills_thinned.png"), thinned_image)
    print(f"[INFO] Images saved in {PATH}")


def main():
    parser = argparse.ArgumentParser(description="Precision Line Art Colorization")
    parser.add_argument(
        "-i", "--image", type=str, required=True, help="Path to input line art image."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="Output directory."
    )
    parser.add_argument(
        "--merge_area",
        type=int,
        default=250,
        help="Max area for a region to be 'small' and merged.",
    )
    parser.add_argument(
        "--erosion",
        type=int,
        default=2,
        help="Erosion iterations for watershed markers (1=fine details, 3=more merging).",
    )
    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    pipeline = ColorizationPipeline(
        min_merge_area=args.merge_area, erosion_iterations=args.erosion
    )
    fillmap = pipeline.process(img_path=args.image)
    saveAll(fillmap, output_path)
    print("[SUCCESS] Processing complete.")


if __name__ == "__main__":
    main()
