import cv2
import numpy as np
import time
import argparse
from pathlib import Path
from scipy.ndimage import distance_transform_edt
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from numba import njit


# --- Logger for verbosity ---
class AxiomLogger:
    def info(self, msg):
        print(f"[AXIOM] {msg}")


logger = AxiomLogger()


# --- Numba-accelerated thinning from WaterShed example, as a utility ---
@njit(nogil=True, fastmath=True)
def fast_thinning(fillmap: np.ndarray, max_iter: int = 100):
    """
    Fills line art pixels (label 0) by iteratively growing neighbor regions.
    This is purely an aesthetic utility for the final render.
    """
    h, w = fillmap.shape
    result = fillmap.copy()

    for _ in range(max_iter):
        # A copy to store updates for the current pass
        temp_result = result.copy()
        line_points_y, line_points_x = np.where(result == 0)

        if line_points_y.shape[0] == 0:
            break

        # For each line pixel, find the first non-zero neighbor and adopt its ID
        for i in range(line_points_y.shape[0]):
            y, x = line_points_y[i], line_points_x[i]
            # 8-way neighbor check
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dx == 0 and dy == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbor_id = result[ny, nx]
                        if neighbor_id != 0:
                            temp_result[y, x] = neighbor_id
                            break  # Found a neighbor, move to next pixel
                if temp_result[y, x] != 0:
                    break

        if np.array_equal(result, temp_result):
            break
        result = temp_result

    return result


# --- The Definitive Segmentation Engine ---
class AxiomV2_Engine:
    def __init__(self, weld_tolerance: int = 5, min_area: int = 250):
        self.weld_tolerance = weld_tolerance
        self.min_area = min_area
        self.original_line_mask = None

    def _read_image(self, img_path: str, threshold: int) -> np.ndarray:
        logger.info(f"Reading image: {img_path}")
        try:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError
            # Lines are 0, background is 255. This is the convention.
            _, self.original_line_mask = cv2.threshold(
                img, threshold, 255, cv2.THRESH_BINARY_INV
            )
            return self.original_line_mask
        except Exception as e:
            raise IOError(f"Could not read or process image at {img_path}: {e}")

    def _get_raw_segmentation(self, line_mask: np.ndarray) -> np.ndarray:
        # Stage 1: Gap Welding
        welded_mask = line_mask
        if self.weld_tolerance > 0:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (self.weld_tolerance, self.weld_tolerance)
            )
            welded_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)

        # Stage 2: Distance Transform
        distance_field, _ = distance_transform_edt(
            welded_mask, return_distances=True, return_indices=False
        )

        # Stage 3: Marker-Controlled Watershed
        # We find seeds far from any line to ensure they are in significant regions.
        # The threshold is based on area, not just distance, for robustness.
        min_dist_from_line = max(2, int(np.sqrt(self.min_area) / np.pi))
        markers_coords = peak_local_max(distance_field, min_distance=min_dist_from_line)
        marker_labels = np.zeros(distance_field.shape, dtype=np.int32)
        for i, (y, x) in enumerate(markers_coords):
            marker_labels[y, x] = i + 1

        logger.info(f"Axiom Core generated {len(markers_coords)} raw segments.")
        return watershed(-distance_field, marker_labels, mask=welded_mask)

    def _semantic_merge(self, raw_map: np.ndarray) -> np.ndarray:
        logger.info("Performing single-pass semantic merge...")
        unique_labels, areas = np.unique(raw_map, return_counts=True)
        area_map = dict(zip(unique_labels, areas))

        # Build adjacency graph - a good piece of engineering from the WaterShed example.
        graph = {}
        right_pairs = np.c_[raw_map[:, :-1].ravel(), raw_map[:, 1:].ravel()]
        down_pairs = np.c_[raw_map[:-1, :].ravel(), raw_map[1:, :].ravel()]
        all_pairs = np.vstack([right_pairs, down_pairs])
        # Filter out self-pairs and pairs involving the line art
        valid_pairs = all_pairs[
            (all_pairs[:, 0] != all_pairs[:, 1])
            & (all_pairs[:, 0] != 0)
            & (all_pairs[:, 1] != 0)
        ]
        sorted_pairs = np.sort(valid_pairs, axis=1)
        unique_edges, counts = np.unique(sorted_pairs, axis=0, return_counts=True)

        # DSU (Disjoint Set Union) data structure for efficient merging
        parent = {int(label): int(label) for label in unique_labels if label != 0}

        # Build a true adjacency list for neighbor lookups
        adj = {label: {} for label in parent}
        for i, edge in enumerate(unique_edges):
            u, v = int(edge[0]), int(edge[1])
            count = int(counts[i])
            adj[u][v] = count
            adj[v][u] = count

        # Rule 1: Containment Merge (merge islands first)
        for label in parent:
            if len(adj.get(label, {})) == 1:
                neighbor = list(adj[label].keys())[0]
                parent[label] = neighbor  # Merge this island

        # Rule 2: Insignificance Merge (sorted by area)
        sorted_labels = sorted(parent.keys(), key=lambda l: area_map.get(l, 0))
        for label in sorted_labels:
            if parent[label] == label and area_map.get(label, 0) < self.min_area:
                best_neighbor, max_border = -1, -1
                for neighbor, border in adj.get(label, {}).items():
                    # Find root of neighbor to handle chained merges
                    root_neighbor = neighbor
                    while parent[root_neighbor] != root_neighbor:
                        root_neighbor = parent[root_neighbor]
                    if root_neighbor != label and border > max_border:
                        max_border = border
                        best_neighbor = root_neighbor
                if best_neighbor != -1:
                    parent[label] = best_neighbor

        # Apply the final mapping in one pass
        final_map = raw_map.copy()
        # Path compression for DSU lookup
        for label_id in sorted(parent.keys()):
            root, path = label_id, [label_id]
            while parent[root] != root:
                root = parent[root]
                path.append(root)
            for node in path:
                parent[node] = root

        # Create the final map from the resolved parents
        # This vectorized approach is faster than a loop
        unique_orig_labels = np.unique(raw_map)
        final_parents = np.array([parent.get(l, l) for l in unique_orig_labels])
        # Use a temporary map for efficient vectorized replacement
        temp_map = np.zeros(unique_orig_labels.max() + 1, dtype=np.int32)
        temp_map[unique_orig_labels] = final_parents

        return temp_map[raw_map]

    def process(self, img_path: str, threshold: int = 220) -> np.ndarray:
        start_time = time.time()

        # 1. Read Image
        initial_mask = self._read_image(img_path, threshold)

        # 2. Get Raw, Perfect Segmentation
        raw_map = self._get_raw_segmentation(initial_mask)

        # 3. Perform Semantic Merging
        merged_map = self._semantic_merge(raw_map)

        # 4. Final Ground-Truth Stamping
        merged_map[self.original_line_mask == 0] = 0

        end_time = time.time()
        logger.info(
            f"Axiom V2 processing complete in {end_time - start_time:.4f} seconds."
        )
        return merged_map


def show_fill_map_final(fillmap: np.ndarray):
    max_label = np.max(fillmap)
    if max_label == 0:
        return np.zeros((*fillmap.shape, 3), dtype=np.uint8)
    np.random.seed(42)  # Deterministic colors for stable visualization
    colors = np.random.randint(64, 255, (int(max_label) + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    return colors[fillmap.astype(int)]


def main():
    parser = argparse.ArgumentParser(
        description="Axiom Core V2 - Definitive Line Art Segmentation"
    )
    parser.add_argument(
        "-i", "--image", type=str, required=True, help="Input line art image."
    )
    parser.add_argument(
        "-o", "--output", type=str, default="output_axiom_v2", help="Output directory."
    )
    parser.add_argument(
        "--weld", type=int, default=5, help="Physical tolerance for welding gaps."
    )
    parser.add_argument(
        "--min_area",
        type=int,
        default=500,
        help="Minimum pixel area for a segment to be kept.",
    )
    parser.add_argument(
        "--thresh",
        type=int,
        default=220,
        help="Luminance threshold for line detection (0-255).",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    engine = AxiomV2_Engine(weld_tolerance=args.weld, min_area=args.min_area)
    final_fillmap = engine.process(img_path=args.image, threshold=args.thresh)

    logger.info("Saving output images...")

    # Save the final data map render
    data_render = show_fill_map_final(final_fillmap)
    cv2.imwrite(
        str(output_path / "fillmap_final.png"),
        cv2.cvtColor(data_render, cv2.COLOR_RGB2BGR),
    )

    # Save the aesthetically pleasing "thinned" version
    thinned_map = fast_thinning(final_fillmap.astype(np.int32))
    thinned_render = show_fill_map_final(thinned_map)
    cv2.imwrite(
        str(output_path / "fillmap_thinned.png"),
        cv2.cvtColor(thinned_render, cv2.COLOR_RGB2BGR),
    )

    logger.info(f"Success. All outputs saved in {output_path}")


if __name__ == "__main__":
    main()
