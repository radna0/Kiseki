import cv2
import numpy as np
import time
import argparse
from pathlib import Path  # Import the Path class
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# Assuming 'thinning' is a pre-existing optimized function
from linefiller.thinning import thinning
from kiseki.logging import logger, Profiler


# --- The Correct Primitive for The Job: A Union-Find Data Structure ---
class UnionFind:
    """A Disjoint-Set Union data structure for managing region merges efficiently."""

    def __init__(self, n):
        # Every region starts as its own parent
        self.parent = np.arange(n, dtype=np.int32)
        # We'll use area as the rank for merging (merge smaller into larger)
        self.area = np.zeros(n, dtype=np.int32)

    def find(self, i):
        # Find the root of i with path compression
        if self.parent[i] == i:
            return i
        self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        # Merge the sets containing i and j
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            # Merge the smaller area region into the larger one
            if self.area[root_i] < self.area[root_j]:
                self.parent[root_i] = root_j
                self.area[root_j] += self.area[root_i]
            else:
                self.parent[root_j] = root_i
                self.area[root_i] += self.area[root_j]
            return True
        return False


class ColorizationPipeline:
    def __init__(self, min_merge_area=150, erosion_iterations=2):
        self.min_merge_area = min_merge_area
        self.erosion_iterations = erosion_iterations

    # --- Unchanged Primitives ---
    def read_image_to_binary(self, img_path: str):
        print("[INFO] Reading image...")
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
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
        return markers.astype(np.int32)

    def heal_and_combine(self, watershed_map: np.ndarray, original_binary: np.ndarray):
        print("[INFO] Healing boundaries and rescuing leftovers...")
        healed_map = watershed_map.copy()
        boundary_mask = healed_map == -1
        # Heal boundaries by replacing with dominant neighbor
        boundary_pixels = np.argwhere(boundary_mask)
        for y, x in boundary_pixels:
            window = healed_map[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
            valid_labels = window[window > 0]
            if valid_labels.size > 0:
                u, c = np.unique(valid_labels, return_counts=True)
                healed_map[y, x] = u[np.argmax(c)]
            else:
                healed_map[y, x] = 0
        # Rescue pass
        found_mask = (healed_map > 0).astype(np.uint8)
        leftovers_mask = cv2.subtract(original_binary, found_mask * 255)
        num_leftovers, leftover_labels = cv2.connectedComponents(leftovers_mask)
        if num_leftovers > 1:
            max_label = np.max(healed_map)
            healed_map[leftover_labels > 0] = (
                leftover_labels[leftover_labels > 0] + max_label
            )
        return healed_map

    def build_region_adjacency_graph(self, labels_map: np.ndarray):
        print("[INFO] Building Region Adjacency Graph...")
        graph = defaultdict(int)
        map_padded = np.pad(labels_map, 1, mode="constant")
        h, w = map_padded.shape
        for y in range(h - 1):
            for x in range(w - 1):
                p1 = map_padded[y, x]
                if p1 == 0:
                    continue
                p2 = map_padded[y, x + 1]
                if p2 != 0 and p1 != p2:
                    graph[tuple(sorted((p1, p2)))] += 1
                p3 = map_padded[y + 1, x]
                if p3 != 0 and p1 != p3:
                    graph[tuple(sorted((p1, p3)))] += 1
        return graph

    # --- The Final Merge Algorithm ---
    def merge_regions_with_union_find(
        self, labels_map: np.ndarray, graph: dict
    ) -> np.ndarray:
        """Merges regions using a Union-Find data structure for correctness and stability."""
        print(f"[INFO] Merging regions with Union-Find structure...")

        max_label_id = np.max(labels_map)
        uf = UnionFind(max_label_id + 1)

        # Populate initial areas for all found labels
        unique_labels, counts = np.unique(labels_map, return_counts=True)
        for label, area in zip(unique_labels, counts):
            if label > 0:
                uf.area[label] = area

        # Iteratively merge until the system is stable
        while True:
            merges_made_in_pass = 0
            # Always process smallest regions first based on current area in UF structure
            # Create a list of current root nodes to process
            root_labels = [l for l in range(1, max_label_id + 1) if uf.parent[l] == l]
            sorted_labels = sorted(root_labels, key=lambda l: uf.area[l])

            for label_id in sorted_labels:
                if uf.area[label_id] < self.min_merge_area:
                    # Find best neighbor based on the static RAG
                    best_neighbor = -1
                    max_border = -1
                    for edge, border_length in graph.items():
                        # Find which part of the edge is our current label and which is the neighbor
                        if label_id in edge:
                            neighbor_id = edge[0] if edge[1] == label_id else edge[1]
                            # Operate on roots to ensure we're comparing final parent regions
                            if uf.find(label_id) != uf.find(neighbor_id):
                                if border_length > max_border:
                                    max_border = border_length
                                    best_neighbor = neighbor_id

                    if best_neighbor != -1:
                        if uf.union(label_id, best_neighbor):
                            merges_made_in_pass += 1

            if merges_made_in_pass == 0:
                print("[INFO] Merge process has converged.")
                break

        print("[INFO] Applying final merge mapping...")

        # --- FIX IS HERE ---
        # The `find` method must be called for each label individually.
        # We build the lookup table with a list comprehension.
        final_mapping = np.array(
            [uf.find(i) for i in range(max_label_id + 1)], dtype=np.int32
        )

        # Use the lookup table to create the final map
        merged_map = final_mapping[labels_map]

        return merged_map

    def reimprint_line_art(self, labels_map: np.ndarray):
        print(
            "[INFO] Finalizing map: Re-imprinting original line art for data integrity..."
        )
        final_map = labels_map.copy()
        line_mask = self.original_binary == 0
        final_map[line_mask] = 0
        return final_map

    def process(self, img_path: str) -> np.ndarray:
        start_time = time.time()
        binary_art = self.read_image_to_binary(img_path)
        watershed_map = self.watershed_segmentation(binary_art)
        combined_map = self.heal_and_combine(watershed_map, binary_art)

        region_graph = self.build_region_adjacency_graph(combined_map)
        merged_map = self.merge_regions_with_union_find(combined_map, region_graph)

        final_map = self.reimprint_line_art(merged_map)
        end_time = time.time()
        print(f"[INFO] Total processing time: {end_time - start_time:.4f} seconds.")
        return final_map


def show_fill_map(fillmap: np.ndarray):
    max_label = np.max(fillmap)
    if max_label == 0:
        return np.zeros((*fillmap.shape, 3), dtype=np.uint8)
    colors = np.random.randint(0, 255, (max_label + 1, 3), dtype=np.uint8)
    colors[0] = [0, 0, 0]
    return colors[fillmap]


def saveAll(fillmap: np.ndarray, PATH: str) -> None:
    """Save results with parallel processing."""
    # Use threading for parallel I/O
    with ThreadPoolExecutor(max_workers=2) as executor:
        # color+undertone
        f1 = show_fill_map(fillmap)
        future1 = executor.submit(cv2.imwrite, PATH / "fills_merged.png", f1)

        # undertone
        f2 = show_fill_map(thinning(fillmap))
        future2 = executor.submit(cv2.imwrite, PATH / "fills_merged_no_contour.png", f2)

        # Wait for completion
        future1.result()
        future2.result()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Data-Preserving Line Art Colorization"
    )

    parser.add_argument(
        "-i", "--image", type=str, required=True, help="Path to input line art image."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",  # Default to a subdirectory
        help="Path to save the output colored map.",
    )
    parser.add_argument(
        "--merge_area",
        type=int,
        default=100,
        help="Max area for a region to be 'small' and merged.",
    )
    parser.add_argument(
        "--erosion",
        type=int,
        default=2,
        help="Erosion iterations for watershed markers (1=fine details, 3=more merging).",
    )

    args = parser.parse_args()
    # --- Pathlib Integration ---
    output_path = Path(args.output)
    # Create the parent directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)

    with Profiler("Trappedball Filling"):
        pipeline = ColorizationPipeline(
            min_merge_area=args.merge_area, erosion_iterations=args.erosion
        )
        fillmap = pipeline.process(img_path=args.image)

    saveAll(fillmap, output_path)
    print(f"[SUCCESS] Output saved to {args.output}")


if __name__ == "__main__":
    main()
