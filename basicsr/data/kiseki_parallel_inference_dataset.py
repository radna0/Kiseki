import os
import os.path as osp
import re
from glob import glob
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import torch.utils.data as data
from natsort import natsorted

from basicsr.utils.registry import DATASET_REGISTRY
from kiseki.paint import read_img_2_np, read_seg_2_np, recolorize_gt, recolorize_seg
from kiseki.logging import logger


@DATASET_REGISTRY.register()
class KisekiParallizedInMemoryInferenceDataset:
    """
    Similar to KisekiInMemoryInferenceDataset but parallelizes the image
    loading + processing snippet via a ThreadPoolExecutor to speed up I/O
    and numpy work. All data is still kept in memory at once (self.samples).
    """

    def __init__(self, opt):
        self.samples = []
        self.opt = opt
        self.root = opt["root"]
        self.multi_clip = opt.get("multi_clip", False)
        self.mode = opt.get("mode", "forward")

        if not self.multi_clip:
            character_paths = [self.root]
        else:
            character_paths = [
                osp.join(self.root, character) for character in os.listdir(self.root)
            ]

        for character_path in character_paths:
            line_root = osp.join(character_path, "line")
            line_list = natsorted(glob(osp.join(line_root, "*.png")))
            logger.info(f"All line List: {line_list}")

            gt_root = osp.join(character_path, "ref")
            gt_list = natsorted(glob(osp.join(gt_root, "*.png")))
            logger.info(f"All REF List: {gt_list}")
            all_gt = [self.convert_gt_path_to_int(gt_path) for gt_path in gt_list]
            logger.info(f"All REF: {all_gt}")

            L = len(line_list)
            # Frame numbers for each line file
            line_frame_numbers = [
                self.convert_gt_path_to_int(line_path) for line_path in line_list
            ]
            logger.info(f"Line Frame Numbers: {line_frame_numbers}")

            # Build index_map: {idx_in_line_list → idx_in_line_list_of_reference}
            if self.mode == "forward":
                index_map = {}
                for idx in range(L):
                    frame_num = line_frame_numbers[idx]
                    if frame_num not in all_gt:
                        # Walk backward until we find a GT
                        prev_idx = idx - 1
                        while (
                            prev_idx >= 0 and line_frame_numbers[prev_idx] not in all_gt
                        ):
                            prev_idx -= 1
                        if prev_idx >= 0:
                            index_map[idx] = prev_idx
                index_list = list(index_map.keys())

            elif self.mode == "nearest":
                # First build a lookup from frame_number → index
                frame_to_idx = {
                    self.convert_gt_path_to_int(path): i
                    for i, path in enumerate(line_list)
                }
                index_map = {}
                for idx in range(L):
                    frame_num = line_frame_numbers[idx]
                    if frame_num not in all_gt:
                        # Find nearest GT frame number
                        nearest_gt = min(all_gt, key=lambda x: abs(x - frame_num))
                        gt_idx = frame_to_idx.get(nearest_gt, None)
                        if gt_idx is None:
                            continue
                        # Move one step toward that GT in the line_list
                        if nearest_gt < frame_num:
                            ref_idx = gt_idx  # can also do gt_idx if you want exact GT
                        else:
                            ref_idx = gt_idx
                        # Ensure ref_idx in bounds
                        if 0 <= ref_idx < L:
                            index_map[idx] = ref_idx
                # topological sort so that chains are processed in order
                index_list = self._sort_indices(index_map)

            elif self.mode in ("reference", "end2end"):
                frame_to_idx = {
                    self.convert_gt_path_to_int(path): i
                    for i, path in enumerate(line_list)
                }
                index_map = {}
                for idx, frame_num in enumerate(line_frame_numbers):
                    if frame_num not in all_gt:
                        nearest_gt = min(all_gt, key=lambda x: abs(x - frame_num))
                        gt_idx = frame_to_idx.get(nearest_gt, None)
                        if gt_idx is None:
                            continue
                        index_map[idx] = gt_idx
                index_list = list(index_map.keys())

            else:
                raise ValueError(f"Unknown mode: {self.mode}")

            logger.info(f"Index list: {index_list}")

            # Prepare a list of arguments for each sample to process in parallel
            items_to_process = []
            for index in index_list:
                # Base line file and its segmentation
                line_path = line_list[index]
                file_name = osp.splitext(osp.basename(line_path))[0]

                seg_path = line_path.replace(
                    os.path.sep + "line" + os.path.sep,
                    os.path.sep + "seg" + os.path.sep,
                )

                # Compute reference index
                ref_idx = index_map[index]
                line_ref_path = line_list[ref_idx]
                file_name_ref = osp.splitext(osp.basename(line_ref_path))[0]

                seg_ref_path = line_ref_path.replace(
                    os.path.sep + "line" + os.path.sep,
                    os.path.sep + "seg" + os.path.sep,
                )

                # Compute the GT path, if it exists
                frame_num_ref = self.convert_gt_path_to_int(line_ref_path)
                if frame_num_ref in all_gt:
                    gt_ref_path = line_ref_path.replace(
                        os.path.sep + "line" + os.path.sep,
                        os.path.sep + "ref" + os.path.sep,
                    )
                else:
                    gt_ref_path = None

                items_to_process.append(
                    (
                        file_name,
                        file_name_ref,
                        line_path,
                        line_ref_path,
                        seg_path,
                        seg_ref_path,
                        gt_ref_path,
                    )
                )

            # Parallelize the actual I/O + numpy + tensor conversion
            num_workers = min(os.cpu_count() or 1, len(items_to_process))
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                for sample_dict in executor.map(self._process_item, items_to_process):
                    if sample_dict is not None:
                        self.samples.append(sample_dict)

            logger.info(f"Length of line frames to be colored: {len(self.samples)}")

    def _process_item(self, args):
        """
        Process a single sample:
            - read images (line, line_ref, seg, seg_ref, optional gt_ref)
            - crop & pad to square
            - extract segmentation stats
            - convert to torch tensors
            - recolor
            - return a dict with all tensors & metadata
        """
        (
            file_name,
            file_name_ref,
            line_path,
            line_ref_path,
            seg_path,
            seg_ref_path,
            gt_ref_path,
        ) = args

        try:
            # 1) Read images from disk
            line_np = read_img_2_np(line_path)
            line_ref_np = read_img_2_np(line_ref_path)

            seg_np = read_seg_2_np(seg_path)
            seg_ref_np = read_seg_2_np(seg_ref_path)

            if gt_ref_path is not None:
                gt_ref_np = read_img_2_np(gt_ref_path)
            else:
                gt_ref_np = None

            # 2) Crop + pad to square
            line_crop, seg_crop, _ = self._square_img_data(line_np, seg_np)
            line_ref_crop, seg_ref_crop, gt_ref_crop = self._square_img_data(
                line_ref_np, seg_ref_np, gt_ref_np
            )

            # 3) Extract segmentation stats
            keypoints, centerpoints, numpixels, seg_relabeled = self._process_seg(
                seg_crop
            )
            keypoints_ref, centerpoints_ref, numpixels_ref, seg_ref_relabeled = (
                self._process_seg(seg_ref_crop)
            )

            # 4) Convert to torch tensors
            line_t = torch.from_numpy(line_crop).permute(2, 0, 1).float() / 255.0
            line_ref_t = (
                torch.from_numpy(line_ref_crop).permute(2, 0, 1).float() / 255.0
            )

            seg_t = torch.from_numpy(seg_relabeled).unsqueeze(0).long()
            seg_ref_t = torch.from_numpy(seg_ref_relabeled).unsqueeze(0).long()

            keypoints_t = torch.from_numpy(keypoints).unsqueeze(0).float()
            keypoints_ref_t = torch.from_numpy(keypoints_ref).unsqueeze(0).float()

            centerpoints_t = torch.from_numpy(centerpoints).unsqueeze(0).float()
            centerpoints_ref_t = torch.from_numpy(centerpoints_ref).unsqueeze(0).float()

            numpixels_t = torch.from_numpy(numpixels).unsqueeze(0).long()
            numpixels_ref_t = torch.from_numpy(numpixels_ref).unsqueeze(0).long()

            # 5) Recolorized image: if we have a GT, recolor that; otherwise recolor from seg
            if gt_ref_crop is None:
                recolorized_np = recolorize_seg(seg_ref_t)
            else:
                recolorized_np = recolorize_gt(gt_ref_crop)

            recolorized_t = recolorized_np.unsqueeze(0)

            return {
                "file_name": file_name,
                "file_name_ref": file_name_ref,
                "keypoints": keypoints_t,
                "keypoints_ref": keypoints_ref_t,
                "centerpoints": centerpoints_t,
                "centerpoints_ref": centerpoints_ref_t,
                "numpixels": numpixels_t,
                "numpixels_ref": numpixels_ref_t,
                "line": line_t.unsqueeze(0),
                "line_ref": line_ref_t.unsqueeze(0),
                "segment": seg_t.unsqueeze(0),
                "segment_ref": seg_ref_t.unsqueeze(0),
                "recolorized_img": recolorized_t,
            }

        except Exception as e:
            logger.warning(f"Skipping sample {file_name} due to error: {e}")
            return None

    def _square_img_data(self, line, seg, gt=None, border=16):
        # Crop to non-background region (white for line, zero for seg), then pad to square.
        mask = np.any(line != [255, 255, 255], axis=-1)
        coords = np.argwhere(mask)
        if coords.size == 0:
            # If the entire image is white, just treat whole image
            y_min, x_min, y_max, x_max = 0, 0, line.shape[0] - 1, line.shape[1] - 1
        else:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

        h, w = line.shape[:2]
        y_min = max(0, y_min - border)
        x_min = max(0, x_min - border)
        y_max = min(h - 1, y_max + border)
        x_max = min(w - 1, x_max + border)

        # Crop
        line_crop = line[y_min : y_max + 1, x_min : x_max + 1]
        seg_crop = seg[y_min : y_max + 1, x_min : x_max + 1]
        gt_crop = None
        if gt is not None:
            gt_crop = gt[y_min : y_max + 1, x_min : x_max + 1]

        # Pad to square
        nh, nw = line_crop.shape[:2]
        diff = abs(nh - nw)
        pad1, pad2 = diff // 2, diff - diff // 2

        if nh > nw:
            # pad left/right
            line_crop = np.pad(
                line_crop, ((0, 0), (pad1, pad2), (0, 0)), constant_values=255
            )
            seg_crop = np.pad(seg_crop, ((0, 0), (pad1, pad2)), constant_values=0)
            if gt_crop is not None:
                gt_crop = np.pad(
                    gt_crop, ((0, 0), (pad1, pad2), (0, 0)), constant_values=0
                )
        else:
            # pad top/bottom
            line_crop = np.pad(
                line_crop, ((pad1, pad2), (0, 0), (0, 0)), constant_values=255
            )
            seg_crop = np.pad(seg_crop, ((pad1, pad2), (0, 0)), constant_values=0)
            if gt_crop is not None:
                gt_crop = np.pad(
                    gt_crop, ((pad1, pad2), (0, 0), (0, 0)), constant_values=0
                )

        return line_crop, seg_crop, gt_crop

    def _process_seg(self, seg):
        """
        Take a 2D segmentation mask (H×W), where 0=background and
        positive integers label components. Return for each component:
          - bounding box [xmin, xmax, ymin, ymax]
          - center [xmean, ymean]
          - pixel count
          - a relabeled segmentation mask in which component IDs run 1..N
        """
        seg_list = np.unique(seg[seg != 0])
        h, w = seg.shape
        hh = np.arange(h)
        ww = np.arange(w)
        xx, yy = np.meshgrid(ww, hh)

        keypoints = []
        centerpoints = []
        numpixels = []
        seg_relabeled = np.zeros_like(seg, dtype=np.int32)

        for i, seg_idx in enumerate(seg_list):
            mask = seg == seg_idx
            xs = xx[mask]
            ys = yy[mask]
            xmin, xmax = xs.min(), xs.max()
            ymin, ymax = ys.min(), ys.max()
            xmean = xs.mean()
            ymean = ys.mean()
            keypoints.append([xmin, xmax, ymin, ymax])
            centerpoints.append([xmean, ymean])
            numpixels.append(int(mask.sum()))
            seg_relabeled[mask] = i + 1  # new IDs start from 1

        if len(keypoints) == 0:
            # No segments detected: return empty arrays
            return (
                np.zeros((0, 4), dtype=np.int32),
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.int32),
                seg_relabeled,
            )

        keypoints = np.stack(keypoints).astype(np.int32)
        centerpoints = np.stack(centerpoints).astype(np.float32)
        numpixels = np.stack(numpixels).astype(np.int32)

        return keypoints, centerpoints, numpixels, seg_relabeled

    def convert_gt_path_to_int(self, gt_path):
        """Extracts trailing digits from filename and returns as int."""
        filename = osp.splitext(osp.basename(gt_path))[0]
        match = re.search(r"(\d+)$", filename)
        if match:
            return int(match.group())
        else:
            raise ValueError(f"No trailing numeric part found in filename: {filename}")

    def _sort_indices(self, index_map):
        """
        Given index_map: {end_idx → start_idx}, produce a deterministic
        topological ordering of `end_idx` keys so dependencies are respected.
        """
        adj_list = defaultdict(list)
        for end_idx, start_idx in index_map.items():
            adj_list[start_idx].append(end_idx)

        visited = set()
        result = []

        def _dfs(node):
            if node not in visited:
                visited.add(node)
                for neighbor in adj_list.get(node, []):
                    _dfs(neighbor)
                result.append(node)

        for node in index_map.keys():
            _dfs(node)

        return result[::-1]
