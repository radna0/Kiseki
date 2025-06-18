import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import defaultdict
from linefiller.thinning import thinning
from kiseki.logging import logger, Profiler


# --- UPGRADE: A proper, high-quality smoothing function ---
def smooth_fill_with_inpainting(fillmap_with_lines: np.ndarray) -> np.ndarray:
    """Creates a smooth, contourless fill by in-painting the line art using a sophisticated algorithm."""
    logger.info(
        "[INFO] Performing high-quality smoothing with Navier-Stokes in-painting..."
    )

    # Create a colored representation of the final map
    colored_view = show_fill_map(fillmap_with_lines, lines_are_black=False)

    # Create a mask of the lines (label 0) which are the areas to be filled
    line_mask = (fillmap_with_lines == 0).astype(np.uint8)

    # Use Navier-Stokes based In-painting. It's computationally more expensive but produces superior results
    # by treating the image as a fluid and propagating color information from the boundaries.
    smoothed_image = cv2.inpaint(colored_view, line_mask, 5, cv2.INPAINT_NS)

    return smoothed_image


class ColorizationPipeline:
    def __init__(self, min_merge_area=250, erosion_iterations=2):
        self.min_merge_area = min_merge_area
        self.erosion_iterations = erosion_iterations

    def read_image_to_binary(self, img_path: str):
        # This function is sound.
        logger.info("[INFO] Reading image...")
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

    def watershed_segmentation(self, line_art_binary: np.ndarray):
        # This function is sound.
        logger.info(
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

    def inpaint_and_resolve_boundaries(self, watershed_map: np.ndarray) -> np.ndarray:
        # This function is sound.
        logger.info(
            "[INFO] Resolving boundaries with Distance Transform + Watershed..."
        )
        markers = watershed_map.copy()
        markers[markers == -1] = 0
        to_fill_mask = (self.original_binary == 0) | (markers == 0)
        dist_transform = cv2.distanceTransform(
            to_fill_mask.astype(np.uint8), cv2.DIST_L2, 5
        )
        cv2.normalize(dist_transform, dist_transform, 0, 255, cv2.NORM_MINMAX)
        dist_8u = dist_transform.astype(np.uint8)
        dist_bgr = cv2.cvtColor(dist_8u, cv2.COLOR_GRAY2BGR)
        cv2.watershed(dist_bgr, markers)
        markers[markers == -1] = 0
        return markers.astype(np.int32)

    def build_region_adjacency_graph(self, labels_map: np.ndarray):
        # This function is sound, but now operates on sanitized data.
        logger.info("[INFO] Building Region Adjacency Graph on sanitized map...")
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
        # This function is sound, but now operates on a correct graph.
        logger.info(
            f"[INFO] Iteratively merging regions smaller than {self.min_merge_area} pixels..."
        )
        unique_labels, counts = np.unique(labels_map, return_counts=True)
        areas = dict(zip(unique_labels, counts))
        parent = {label: label for label in unique_labels if label > 0}

        if not parent:
            return labels_map

        pass_num = 0
        while True:
            pass_num += 1
            merges_made_this_pass = 0
            sorted_labels = sorted(
                [l for l in parent if parent[l] == l and l in areas],
                key=lambda l: areas[l],
            )
            for label_id in sorted_labels:
                if areas.get(label_id, 0) < self.min_merge_area:
                    best_neighbor, max_border = -1, -1
                    for edge, border_length in graph.items():
                        if label_id in edge:
                            neighbor_id = edge[0] if edge[1] == label_id else edge[1]
                            root_neighbor = neighbor_id
                            if root_neighbor not in parent:
                                continue
                            while parent[root_neighbor] != root_neighbor:
                                root_neighbor = parent[root_neighbor]
                            if label_id != root_neighbor and border_length > max_border:
                                max_border, best_neighbor = border_length, root_neighbor
                    if best_neighbor != -1:
                        parent[label_id] = best_neighbor
                        areas[best_neighbor] = areas.get(best_neighbor, 0) + areas.get(
                            label_id, 0
                        )
                        areas.pop(label_id, None)
                        merges_made_this_pass += 1
            if merges_made_this_pass == 0:
                logger.info(f"[INFO] Merge process converged after {pass_num} passes.")
                break

        logger.info("[INFO] Applying final merge mapping...")
        final_map = np.zeros_like(labels_map)
        for label_id in sorted(parent.keys()):
            root = label_id
            path_to_root = [root]
            while parent[root] != root:
                root = parent[root]
                path_to_root.append(root)
            for node in path_to_root:
                parent[node] = root
            final_map[labels_map == label_id] = root

        return final_map

    def process(self, img_path: str) -> np.ndarray:
        start_time = time.time()
        binary_art = self.read_image_to_binary(img_path)

        # 1. Core segmentation
        watershed_map = self.watershed_segmentation(binary_art)

        # 2. High-quality boundary and leftover resolution
        resolved_map = self.inpaint_and_resolve_boundaries(watershed_map)

        # 3. --- The Critical Philosophy Change ---
        # Sanitize the map BEFORE building the graph to ensure topological correctness.
        sanitized_map = resolved_map.copy()
        sanitized_map[self.original_binary == 0] = 0

        # 4. Intelligent merging based on a PERFECT graph
        region_graph = self.build_region_adjacency_graph(sanitized_map)
        merged_map = self.merge_regions_iteratively(sanitized_map, region_graph)

        # 5. Final data integrity pass to ensure no merge operations bled over lines
        final_map = merged_map.copy()
        final_map[self.original_binary == 0] = 0

        end_time = time.time()
        logger.info(
            f"[INFO] Total processing time: {end_time - start_time:.4f} seconds."
        )
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
    logger.info("[INFO] Saving output images...")
    # Save the primary data map render (with lines)
    cv2.imwrite(
        str(PATH / "fills_with_lines.png"), show_fill_map(fillmap, lines_are_black=True)
    )
    # Create and save the "smooth" version using high-quality in-painting
    smoothed_image = smooth_fill_with_inpainting(fillmap)
    cv2.imwrite(str(PATH / "fills_smooth.png"), smoothed_image)

    thinned_image = show_fill_map(thinning(fillmap))
    cv2.imwrite(str(PATH / "fills_thinned.png"), thinned_image)
    logger.info(f"[INFO] Images saved in {PATH}")


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

    with Profiler("Trappedball Filling"):
        pipeline = ColorizationPipeline(
            min_merge_area=args.merge_area, erosion_iterations=args.erosion
        )
        fillmap = pipeline.process(img_path=args.image)
    saveAll(fillmap, output_path)
    logger.info("[SUCCESS] Processing complete.")


if __name__ == "__main__":
    main()
