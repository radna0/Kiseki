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


@njit(nogil=True, fastmath=True)
def build_graph_numba(labels_map: np.ndarray):
    """Builds the RAG with explicit, compiler-friendly logic."""
    graph_counts = {}
    h, w = labels_map.shape
    for y in range(h - 1):
        for x in range(w - 1):
            p1 = labels_map[y, x]
            if p1 == 0:
                continue
            p2 = labels_map[y, x + 1]
            if p2 != 0 and p1 != p2:
                edge = (p1, p2) if p1 < p2 else (p2, p1)
                if edge in graph_counts:
                    graph_counts[edge] += 1
                else:
                    graph_counts[edge] = 1
            p3 = labels_map[y + 1, x]
            if p3 != 0 and p1 != p3:
                edge = (p1, p3) if p1 < p3 else (p3, p1)
                if edge in graph_counts:
                    graph_counts[edge] += 1
                else:
                    graph_counts[edge] = 1
    return graph_counts


class ColorizationPipeline:
    def __init__(self, min_merge_area=500, noise_threshold=50):
        self.min_merge_area = min_merge_area
        self.noise_threshold = noise_threshold

    def read_image_to_binary(self, img_path: str):
        """Reads image and creates an unmodified binary line art mask."""
        print("[INFO] Reading image...")
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Image not found at {img_path}")
        if img.shape[2] == 4:
            source_channel = img[:, :, 3]
            _, binary_img = cv2.threshold(source_channel, 128, 255, cv2.THRESH_BINARY)
        else:
            source_channel = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary_img = cv2.threshold(
                source_channel, 220, 255, cv2.THRESH_BINARY_INV
            )
        self.original_line_mask = binary_img == 255
        return cv2.bitwise_not(binary_img)

    def merge_regions_iteratively(
        self, labels_map: np.ndarray, stats: np.ndarray, graph: dict
    ):
        """A corrected, stable, iterative merge process."""
        print(
            f"[INFO] Iteratively merging regions smaller than {self.min_merge_area} pixels..."
        )
        areas = {
            label: stats[label, cv2.CC_STAT_AREA] for label in range(1, stats.shape[0])
        }
        parent = {label: label for label in areas}
        if not parent:
            return labels_map
        pass_num = 0
        while True:
            pass_num += 1
            merges_made_this_pass = 0
            root_labels = [l for l in parent if parent[l] == l]
            sorted_labels = sorted(root_labels, key=lambda l: areas.get(l, 0))
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
                        area1 = areas.get(label_id, 0)
                        area2 = areas.get(best_neighbor, 0)
                        if area1 < area2:
                            parent[label_id] = best_neighbor
                            areas[best_neighbor] = area1 + area2
                        else:
                            parent[best_neighbor] = label_id
                            areas[label_id] = area1 + area2
                        areas.pop(label_id, None)
                        merges_made_this_pass += 1
            if merges_made_this_pass == 0:
                print(f"[INFO] Merge process converged after {pass_num} passes.")
                break

        remap_arr = np.arange(labels_map.max() + 1, dtype=np.int32)
        for label_id in parent:
            root, path_to_root = label_id, [label_id]
            while parent[root] != root:
                root = parent[root]
                path_to_root.append(root)
            for node in path_to_root:
                parent[node] = root
            remap_arr[label_id] = root
        return remap_arr[labels_map]

    def despeckle(self, labels_map: np.ndarray) -> np.ndarray:
        """Final pass to remove any tiny noise artifacts."""
        print(
            f"[INFO] Despeckle: Removing noise smaller than {self.noise_threshold} pixels..."
        )
        output_map = labels_map.copy()
        unique_labels, counts = np.unique(output_map, return_counts=True)
        for label, count in zip(unique_labels, counts):
            if label > 0 and count < self.noise_threshold:
                output_map[output_map == label] = 0
        return output_map

    def process(self, img_path: str) -> np.ndarray:
        start_time = time.time()
        binary_art = self.read_image_to_binary(img_path)

        # --- The Definitive Multi-Scale Difference Segmentation Engine ---
        print("[INFO] Starting Multi-Scale Difference Segmentation...")

        num_labels, final_map, stats, _ = cv2.connectedComponentsWithStats(
            binary_art, connectivity=8, ltype=cv2.CV_32S
        )

        radii = [3, 7, 15]
        inverted_binary = cv2.bitwise_not(binary_art)

        for r in radii:
            print(f"[INFO] Processing scale with radius: {r}")
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)
            )
            closed_inverted = cv2.morphologyEx(inverted_binary, cv2.MORPH_CLOSE, kernel)
            closed_binary = cv2.bitwise_not(closed_inverted)

            num_labels_new, labels_map_new, _, _ = cv2.connectedComponentsWithStats(
                closed_binary, connectivity=8, ltype=cv2.CV_32S
            )

            difference_mask = (final_map == 0) & (labels_map_new > 0)

            if np.any(difference_mask):
                # Find the maximum label in the current map to avoid collisions
                max_current_label = np.max(final_map)
                new_labels = labels_map_new[difference_mask]
                # Remap new labels to be unique
                unique_new = np.unique(new_labels)
                remap_table = np.zeros(unique_new.max() + 1, dtype=np.int32)
                for new_l in unique_new:
                    max_current_label += 1
                    remap_table[new_l] = max_current_label
                # Apply remapped labels to the final map
                final_map[difference_mask] = remap_table[new_labels]

        print(
            "[INFO] Multi-scale segmentation complete. Now beginning post-processing."
        )

        # Re-run connected components to get a clean final set of labels and stats
        num_labels, final_map, stats, _ = cv2.connectedComponentsWithStats(
            (final_map > 0).astype(np.uint8), connectivity=8, ltype=cv2.CV_32S
        )

        region_graph = build_graph_numba(final_map)
        merged_map = self.merge_regions_iteratively(final_map, stats, region_graph)
        despeckled_map = self.despeckle(merged_map)

        final_map = despeckled_map.copy()
        final_map[self.original_line_mask] = 0

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
        default=50,
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
