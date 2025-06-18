import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from collections import defaultdict
from linefiller.thinning import thinning
from numba import njit
from kiseki.logging import logger, Profiler


@njit(nogil=True, fastmath=True)
def fast_modal_thinning(fillmap: np.ndarray, max_iter: int = 15):
    """Fills line art pixels by finding the MODE of its valid neighbors for a sharp, non-biased fill."""
    h, w = fillmap.shape
    result = fillmap.copy()
    neighbors = np.zeros(8, dtype=result.dtype)
    for _ in range(max_iter):
        changed_in_pass = False
        line_points_y, line_points_x = np.where(result == 0)
        if line_points_y.shape[0] == 0:
            break
        for i in range(line_points_y.shape[0]):
            y, x = line_points_y[i], line_points_x[i]
            n_count = 0
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
                max_freq, mode = 0, -1
                for k in range(n_count):
                    freq = 1
                    for l in range(k + 1, n_count):
                        if neighbors[k] == neighbors[l]:
                            freq += 1
                    if freq > max_freq:
                        max_freq, mode = freq, neighbors[k]
                if mode != -1 and result[y, x] != mode:
                    result[y, x] = mode
                    changed_in_pass = True
        if not changed_in_pass:
            break
    return result


class ColorizationPipeline:
    def __init__(self, min_merge_area=500, noise_threshold=20):
        self.min_merge_area = min_merge_area
        self.noise_threshold = noise_threshold

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
        closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=1)
        self.authoritative_line_mask = cv2.dilate(closed, kernel, iterations=1) == 255
        self.original_binary = np.where(self.authoritative_line_mask, 0, 255).astype(
            np.uint8
        )
        return self.original_binary

    # --- UPGRADE: Guaranteed Marker Generation ---
    def watershed_segmentation(self, line_art_binary: np.ndarray):
        print("[INFO] Performing Watershed with Guaranteed Markers...")

        # 1. Find ALL potential regions first to guarantee no region is lost.
        num_components, all_components, stats, centroids = (
            cv2.connectedComponentsWithStats(line_art_binary)
        )

        # 2. Create the marker map from these components.
        # We add 1 because watershed uses 0 as a boundary.
        markers = all_components.astype(np.int32) + 1

        # 3. Mark the line art area as the 'unknown' region for watershed to solve.
        markers[self.authoritative_line_mask] = 0

        # 4. Run the watershed.
        markers = cv2.watershed(self.original_bgr, markers)
        markers[markers == -1] = 0  # Final boundaries become lines
        return markers

    def build_region_adjacency_graph(self, labels_map: np.ndarray):
        """Builds a graph using a generous 3x3 neighborhood search to ensure all adjacencies are found."""
        print("[INFO] Building topologically-generous Region Adjacency Graph...")
        # This function is JIT-compiled for extreme performance
        return build_graph_numba(labels_map)

    def merge_regions_iteratively(self, labels_map: np.ndarray, graph: dict):
        # This function is now fed a perfect graph.
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
                            root_neighbor = parent[neighbor_id]
                            while parent[root_neighbor] != root_neighbor:
                                root_neighbor = parent[root_neighbor]
                            if label_id != root_neighbor and border_length > max_border:
                                max_border, best_neighbor = border_length, root_neighbor
                    if best_neighbor != -1:
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

        remap_arr = np.zeros(labels_map.max() + 1, dtype=np.int32)
        for label_id in parent:
            root = parent[label_id]
            while parent[root] != root:
                root = parent[root]
            remap_arr[label_id] = root
        return remap_arr[labels_map]

    # --- UPGRADE: Final, absolute denoising pass ---
    def despeckle(self, labels_map: np.ndarray) -> np.ndarray:
        """Removes any remaining small noise artifacts after all merging is complete."""
        print(
            f"[INFO] Despeckle: Removing noise regions smaller than {self.noise_threshold} pixels..."
        )
        output_map = labels_map.copy()
        unique_labels, counts = np.unique(output_map, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label > 0 and count < self.noise_threshold:
                output_map[output_map == label] = 0  # Erase noise to line art color
        return output_map

    def process(self, img_path: str) -> np.ndarray:
        start_time = time.time()
        binary_art = self.create_authoritative_line_mask(img_path)
        resolved_map = self.watershed_segmentation(binary_art)
        region_graph = self.build_region_adjacency_graph(resolved_map)
        merged_map = self.merge_regions_iteratively(resolved_map, region_graph)
        despeckled_map = self.despeckle(merged_map)
        final_map = despeckled_map.copy()
        final_map[self.authoritative_line_mask] = 0
        end_time = time.time()
        print(f"[INFO] Total processing time: {end_time - start_time:.4f} seconds.")
        return final_map


@njit(nogil=True, fastmath=True)
def build_graph_numba(labels_map: np.ndarray):
    """Numba-accelerated function to build the RAG with explicit, compiler-friendly logic."""
    graph_counts = {}  # Standard dict is fine if we are explicit below
    h, w = labels_map.shape
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            p_center = labels_map[y, x]
            if p_center == 0:
                continue

            # Check 3x3 window around the center pixel
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue

                    p_neighbor = labels_map[y + dy, x + dx]

                    # Add an edge if we find a different, valid region
                    if p_neighbor != 0 and p_center != p_neighbor:
                        edge = (
                            (p_center, p_neighbor)
                            if p_center < p_neighbor
                            else (p_neighbor, p_center)
                        )

                        # --- FIX IS HERE ---
                        # Replace the ambiguous .get() with an explicit if/else block.
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
        "--noise",
        type=int,
        default=25,
        help="Pixel area to consider as noise to be removed.",
    )

    args = parser.parse_args()
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    with Profiler("Trappedball Filling"):
        pipeline = ColorizationPipeline(
            min_merge_area=args.merge_area, noise_threshold=args.noise
        )
        fillmap = pipeline.process(img_path=args.image)
    saveAll(fillmap, output_path)
    logger.info("[SUCCESS] Processing complete.")


if __name__ == "__main__":
    main()
