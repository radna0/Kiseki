import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import defaultdict
from linefiller.thinning import thinning
from numba import njit
from kiseki.logging import logger, Profiler


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


class ColorizationPipeline:
    def __init__(self, min_merge_area=250, erosion_iterations=2):
        self.min_merge_area = min_merge_area
        self.erosion_iterations = erosion_iterations

    def read_image_to_binary(self, img_path: str):
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

    def watershed_segmentation(self, line_art_binary: np.ndarray):
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
        markers[markers == -1] = 0  # Treat boundaries as lines
        return markers.astype(np.int32)

    def build_region_adjacency_graph(self, labels_map: np.ndarray):
        print("[INFO] Building Region Adjacency Graph...")
        graph = defaultdict(int)
        h, w = labels_map.shape
        # This is a faster way to build the graph than iterating every pixel
        # Check horizontal neighbors
        right_diff = labels_map[:, 1:] != labels_map[:, :-1]
        right_pairs = np.c_[
            labels_map[:, :-1][right_diff], labels_map[:, 1:][right_diff]
        ]
        # Check vertical neighbors
        down_diff = labels_map[1:, :] != labels_map[:-1, :]
        down_pairs = np.c_[labels_map[:-1, :][down_diff], labels_map[1:, :][down_diff]]

        all_pairs = np.vstack([right_pairs, down_pairs])

        # Filter out pairs involving line art (0) and sort for canonical key
        valid_pairs = all_pairs[(all_pairs[:, 0] != 0) & (all_pairs[:, 1] != 0)]
        valid_pairs.sort(axis=1)

        # Count occurrences of each unique pair
        unique_edges, counts = np.unique(valid_pairs, axis=0, return_counts=True)
        for i in range(unique_edges.shape[0]):
            graph[tuple(unique_edges[i])] = counts[i]

        return graph

    def merge_regions_iteratively(self, labels_map: np.ndarray, graph: dict):
        print(
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
                print(f"[INFO] Merge process converged after {pass_num} passes.")
                break

        print("[INFO] Applying final merge mapping...")
        final_map = np.zeros_like(labels_map)
        for label_id in sorted(parent.keys()):
            root, path_to_root = label_id, [label_id]
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

        # 1. Core segmentation - gives raw regions and boundaries
        resolved_map = self.watershed_segmentation(binary_art)

        # 2. Sanitize the map with ground-truth line art BEFORE any merging.
        # This is the critical step to prevent spills.
        sanitized_map = resolved_map.copy()
        sanitized_map[self.original_binary == 0] = 0

        # 3. Intelligent merging based on a now-perfect graph
        region_graph = self.build_region_adjacency_graph(sanitized_map)
        merged_map = self.merge_regions_iteratively(sanitized_map, region_graph)

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
    logger.info("[INFO] Saving output images...")
    # Save the primary data map render (with lines)
    cv2.imwrite(
        str(PATH / "fills_with_lines.png"), show_fill_map(fillmap, lines_are_black=True)
    )
    # Create and save the "smooth" version using high-quality in-painting
    smoothed_image = show_fill_map(fast_thinning(fillmap))
    cv2.imwrite(str(PATH / "fills_fast_thinned.png"), smoothed_image)

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
        default=1,
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
