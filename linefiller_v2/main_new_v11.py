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
    def __init__(self, min_merge_area=350, erosion_iterations=2):
        self.min_merge_area = min_merge_area
        self.erosion_iterations = erosion_iterations

    # --- UPGRADE: Create a definitive, slightly thickened line mask ---
    def create_authoritative_line_mask(self, img_path: str):
        """Creates a robust line mask that captures the core line and its anti-aliasing."""
        print("[INFO] Creating authoritative line mask...")
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
            # Use a low threshold to catch faint anti-aliasing pixels
            _, binary_img = cv2.threshold(source_channel, 1, 255, cv2.THRESH_BINARY)
        else:
            source_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(
                source_channel, 250, 255, cv2.THRESH_BINARY_INV
            )

        # Dilate the mask slightly to ensure it covers all anti-aliasing.
        # This creates our absolute "no-spill" zone.
        kernel = np.ones((3, 3), np.uint8)
        self.authoritative_line_mask = (
            cv2.dilate(binary_img, kernel, iterations=1) == 255
        )

        # The binary for watershed is the inverse of this authoritative mask
        self.original_binary = cv2.bitwise_not(
            cv2.UMat(self.authoritative_line_mask.astype(np.uint8) * 255)
        ).get()
        return self.original_binary

    def watershed_segmentation(self, line_art_binary: np.ndarray):
        # This function is now simpler as it just produces the raw segmentation
        print(
            f"[INFO] Performing Watershed... (Erosion: {self.erosion_iterations} iter)"
        )
        kernel = np.ones((3, 3), np.uint8)
        sure_fg = cv2.erode(line_art_binary, kernel, iterations=self.erosion_iterations)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[cv2.dilate(line_art_binary, kernel, iterations=3) == 0] = 0
        markers = cv2.watershed(self.original_bgr, markers)
        return markers.astype(np.int32)

    # --- FIX: Replaced the entire function with a robust, Numba-accelerated version ---
    def build_region_adjacency_graph(self, labels_map: np.ndarray):
        """Builds a graph of adjacent regions and their shared border lengths."""
        print("[INFO] Building Region Adjacency Graph...")
        # This function is now JIT-compiled for performance
        return build_graph_numba(labels_map)

    def merge_regions_iteratively(self, labels_map: np.ndarray, graph: dict):
        # This logic is sound, but now receives a perfect graph.
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
        final_mapping = {label: parent[label] for label in parent}
        for label_id in sorted(parent.keys()):
            root = label_id
            path_to_root = [root]
            while final_mapping[root] != root:
                root = final_mapping[root]
                path_to_root.append(root)
            for node in path_to_root:
                final_mapping[node] = root

        unique_original_labels = np.unique(labels_map)
        remap_arr = np.zeros(unique_original_labels.max() + 1, dtype=np.int32)
        for l in unique_original_labels:
            if l == 0:
                remap_arr[l] = 0
            else:
                remap_arr[l] = final_mapping.get(l, l)

        return remap_arr[labels_map]

    def process(self, img_path: str) -> np.ndarray:
        start_time = time.time()

        # 1. Create the authoritative line mask. This is our ground truth.
        binary_art = self.create_authoritative_line_mask(img_path)

        # 2. Core segmentation
        resolved_map = self.watershed_segmentation(binary_art)
        resolved_map[resolved_map == -1] = 0  # Set watershed boundaries to line color

        # 3. Build graph on a 'logic map' to ensure adjacencies are found.
        region_graph = self.build_region_adjacency_graph(resolved_map)

        # 4. Merge regions on the clean 'paint map' using decisions from the perfect graph.
        merged_map = self.merge_regions_iteratively(resolved_map, region_graph)

        # 5. The final, absolute enforcement of the ground truth line art.
        final_map = merged_map.copy()
        final_map[self.authoritative_line_mask] = 0

        end_time = time.time()
        print(f"[INFO] Total processing time: {end_time - start_time:.4f} seconds.")
        return final_map


# --- The new Numba-accelerated graph builder, defined outside the class ---
@njit(nogil=True, fastmath=True)
def build_graph_numba(labels_map: np.ndarray):
    """Numba-accelerated function to build the RAG."""
    graph_counts = {}  # Numba works better with dicts than defaultdict
    h, w = labels_map.shape
    for y in range(h):
        for x in range(w):
            p1 = labels_map[y, x]
            if p1 == 0:
                continue

            # Check right neighbor
            if x + 1 < w:
                p2 = labels_map[y, x + 1]
                if p2 != 0 and p1 != p2:
                    # --- FIX IS HERE ---
                    # Replace tuple(sorted(...)) with a direct conditional expression
                    # to create the canonical tuple key. This is Numba-friendly.
                    edge = (p1, p2) if p1 < p2 else (p2, p1)
                    graph_counts[edge] = graph_counts.get(edge, 0) + 1

            # Check bottom neighbor
            if y + 1 < h:
                p3 = labels_map[y + 1, x]
                if p3 != 0 and p1 != p3:
                    # --- FIX IS HERE ---
                    edge = (p1, p3) if p1 < p3 else (p3, p1)
                    graph_counts[edge] = graph_counts.get(edge, 0) + 1
    return graph_counts


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
        default=500,
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
