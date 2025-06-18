import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import defaultdict
from linefiller.thinning import thinning
from numba import njit
from kiseki.logging import logger, Profiler


# --- UPGRADE: The definitive, high-quality, high-speed thinning function ---
@njit(nogil=True, fastmath=True)
def fast_modal_thinning(fillmap: np.ndarray, max_iter: int = 15):
    """
    Fills line art pixels (label 0) by finding the MODE of its valid neighbors.
    This is the logically correct approach for sharp, non-biased fills.
    """
    h, w = fillmap.shape
    result = fillmap.copy()
    # Pre-allocate a small array for neighbor analysis
    neighbors = np.zeros(8, dtype=result.dtype)

    for _ in range(max_iter):
        changed_in_pass = False
        line_points_y, line_points_x = np.where(result == 0)

        if line_points_y.shape[0] == 0:
            break

        for i in range(line_points_y.shape[0]):
            y, x = line_points_y[i], line_points_x[i]

            # Collect valid neighbors
            n_count = 0
            # Manually unrolled loop for 8-way check
            if y > 0 and result[y - 1, x] != 0:
                neighbors[n_count] = result[y - 1, x]
                n_count += 1
            if y < h - 1 and result[y + 1, x] != 0:
                neighbors[n_count] = result[y + 1, x]
                n_count += 1
            if x > 0 and result[y, x - 1] != 0:
                neighbors[n_count] = result[y, x - 1]
                n_count += 1
            if x < w - 1 and result[y, x + 1] != 0:
                neighbors[n_count] = result[y, x + 1]
                n_count += 1
            if y > 0 and x > 0 and result[y - 1, x - 1] != 0:
                neighbors[n_count] = result[y - 1, x - 1]
                n_count += 1
            if y > 0 and x < w - 1 and result[y - 1, x + 1] != 0:
                neighbors[n_count] = result[y - 1, x + 1]
                n_count += 1
            if y < h - 1 and x > 0 and result[y + 1, x - 1] != 0:
                neighbors[n_count] = result[y + 1, x - 1]
                n_count += 1
            if y < h - 1 and x < w - 1 and result[y + 1, x + 1] != 0:
                neighbors[n_count] = result[y + 1, x + 1]
                n_count += 1

            if n_count > 0:
                # Find the mode (most frequent) of the collected neighbors
                max_freq = 0
                mode = -1
                for k in range(n_count):
                    freq = 1
                    for l in range(k + 1, n_count):
                        if neighbors[k] == neighbors[l]:
                            freq += 1
                    if freq > max_freq:
                        max_freq = freq
                        mode = neighbors[k]

                if mode != -1 and result[y, x] != mode:
                    result[y, x] = mode
                    changed_in_pass = True

        if not changed_in_pass:
            break

    return result


class ColorizationPipeline:
    def __init__(self, min_merge_area=500, erosion_iterations=2):
        self.min_merge_area = min_merge_area
        self.erosion_iterations = erosion_iterations

    def create_authoritative_line_mask(self, img_path: str):
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
            _, binary_img = cv2.threshold(source_channel, 1, 255, cv2.THRESH_BINARY)
        else:
            source_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(
                source_channel, 250, 255, cv2.THRESH_BINARY_INV
            )
        kernel = np.ones((3, 3), np.uint8)
        # Closing fills small holes in lines before dilating
        closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        self.authoritative_line_mask = cv2.dilate(closed, kernel, iterations=1) == 255
        self.original_binary = np.where(self.authoritative_line_mask, 0, 255).astype(
            np.uint8
        )
        return self.original_binary

    def watershed_segmentation(self, line_art_binary: np.ndarray):
        print(
            f"[INFO] Performing Watershed... (Erosion: {self.erosion_iterations} iter)"
        )
        kernel = np.ones((3, 3), np.uint8)
        sure_fg = cv2.erode(line_art_binary, kernel, iterations=self.erosion_iterations)
        ret, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[cv2.dilate(line_art_binary, kernel, iterations=3) == 0] = 0
        markers = cv2.watershed(self.original_bgr, markers)
        markers[markers == -1] = 0
        return markers.astype(np.int32)

    # --- UPGRADE: Unambiguous Numba-compatible graph building ---
    def build_region_adjacency_graph(self, labels_map: np.ndarray):
        print("[INFO] Building unambiguous Region Adjacency Graph...")
        # JIT compile the graph building for extreme performance
        graph_dict = build_graph_numba(labels_map)
        return graph_dict

    def merge_regions_iteratively(self, labels_map: np.ndarray, graph: dict):
        print(
            f"[INFO] Iteratively merging regions smaller than {self.min_merge_area} pixels..."
        )
        unique_labels, counts = np.unique(labels_map, return_counts=True)
        areas = {
            label: count for label, count in zip(unique_labels, counts) if label > 0
        }
        parent = {label: label for label in areas}

        if not parent:
            return labels_map

        pass_num = 0
        while True:
            pass_num += 1
            merges_made_this_pass = 0
            sorted_labels = sorted(
                [l for l in parent if parent[l] == l], key=lambda l: areas.get(l, 0)
            )

            for label_id in sorted_labels:
                if areas.get(label_id, 0) < self.min_merge_area:
                    best_neighbor, max_border = -1, -1
                    for edge, border_length in graph.items():
                        if label_id in edge:
                            neighbor_id = edge[0] if edge[1] == label_id else edge[1]
                            if neighbor_id not in parent:
                                continue

                            # Find ultimate parent of neighbor
                            root_neighbor = parent[neighbor_id]
                            while parent[root_neighbor] != root_neighbor:
                                root_neighbor = parent[root_neighbor]

                            if label_id != root_neighbor and border_length > max_border:
                                max_border, best_neighbor = border_length, root_neighbor

                    if best_neighbor != -1:
                        # Union: merge smaller into larger
                        current_area = areas.get(label_id, 0)
                        neighbor_area = areas.get(best_neighbor, 0)
                        if current_area < neighbor_area:
                            parent[label_id] = best_neighbor
                            areas[best_neighbor] = neighbor_area + current_area
                        else:
                            parent[best_neighbor] = label_id
                            areas[label_id] = neighbor_area + current_area
                        areas.pop(label_id, None)
                        merges_made_this_pass += 1

            if merges_made_this_pass == 0:
                print(f"[INFO] Merge process converged after {pass_num} passes.")
                break

        print("[INFO] Applying final merge mapping...")
        remap_arr = np.zeros(labels_map.max() + 1, dtype=np.int32)
        for label_id in parent:
            root = label_id
            path_to_root = [root]
            while parent[root] != root:
                root = parent[root]
                path_to_root.append(root)
            for node in path_to_root:
                parent[node] = root
            remap_arr[label_id] = root

        return remap_arr[labels_map]

    def process(self, img_path: str) -> np.ndarray:
        start_time = time.time()
        binary_art = self.create_authoritative_line_mask(img_path)
        resolved_map = self.watershed_segmentation(binary_art)

        # Sanitize with the authoritative mask BEFORE any analysis
        sanitized_map = resolved_map.copy()
        sanitized_map[self.authoritative_line_mask] = 0

        region_graph = self.build_region_adjacency_graph(sanitized_map)
        merged_map = self.merge_regions_iteratively(sanitized_map, region_graph)

        final_map = merged_map.copy()
        final_map[self.authoritative_line_mask] = 0

        end_time = time.time()
        print(f"[INFO] Total processing time: {end_time - start_time:.4f} seconds.")
        return final_map


@njit(nogil=True, fastmath=True)
def build_graph_numba(labels_map: np.ndarray):
    """Numba-accelerated function to build the RAG with explicit, compiler-friendly logic."""
    graph_counts = {}  # Standard dict is fine if we are explicit below
    h, w = labels_map.shape
    for y in range(h - 1):
        for x in range(w - 1):
            p1 = labels_map[y, x]
            if p1 == 0:
                continue

            # Check right neighbor
            p2 = labels_map[y, x + 1]
            if p2 != 0 and p1 != p2:
                edge = (p1, p2) if p1 < p2 else (p2, p1)
                # --- FIX IS HERE ---
                # Replace the ambiguous .get() with an explicit if/else block.
                if edge in graph_counts:
                    graph_counts[edge] += 1
                else:
                    graph_counts[edge] = 1

            # Check bottom neighbor
            p3 = labels_map[y + 1, x]
            if p3 != 0 and p1 != p3:
                edge = (p1, p3) if p1 < p3 else (p3, p1)
                # --- FIX IS HERE ---
                if edge in graph_counts:
                    graph_counts[edge] += 1
                else:
                    graph_counts[edge] = 1
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
    smoothed_image = show_fill_map(fast_modal_thinning(fillmap))
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
