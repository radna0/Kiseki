import re
import numpy as np

import os
import os.path as osp
import cv2
import torch
import torch.utils.data as data
from collections import defaultdict
from glob import glob

from basicsr.utils.registry import DATASET_REGISTRY
from kiseki.paint import read_img_2_np, read_seg_2_np, recolorize_gt, recolorize_seg
from natsort import natsorted
from numba import njit, prange


from numba import njit, prange
import numpy as np

import sys

sys.path.append("...")  # Adds higher directory to python modules path.
from kiseki.logging import logger

RAFT_RESOLUTIONS = [(1280, 720), (1024, 1024), (720, 1280)]


@njit
def _process_seg(seg):
    """
    Numba‐accelerated version of `_process_seg`.  Assumes:
    - seg is a 2D integer array of shape (H, W)
    - labels run from 0..L (with 0 meaning “background”)
    - we want to produce:
        keypoints    : (num_labels, 4)  → [xmin, xmax, ymin, ymax]
        centerpoints : (num_labels, 2)  → [xmean, ymean]
        numpixels    : (num_labels,)    → pixel count
        seg_relabeled: same shape as seg, but all >0 pixels re‐labeled 1..num_labels
    """
    H, W = seg.shape

    # 1) Find the maximum nonzero label so we can size our arrays
    max_label = 0
    for i in range(H):
        for j in range(W):
            v = seg[i, j]
            if v > max_label:
                max_label = v

    if max_label == 0:
        # No segments at all → return empty structures
        return (
            np.zeros((0, 4), np.int32),
            np.zeros((0, 2), np.float32),
            np.zeros((0,), np.int32),
            np.zeros_like(seg, np.int32),
        )

    # 2) We know labels run 1..max_label.  We'll build stats for each label index (0..max_label-1)
    L = max_label

    # Initialize bounding‐box trackers, sum‐of‐coords, and counts
    #   - bbox_min_x[k] will track the minimum x‐coordinate of any pixel whose seg == (k+1)
    #   - bbox_max_x[k] will track the maximum x‐coordinate of the same
    #   - Likewise for y
    inf = 10**9
    bbox_min_x = np.full(L, inf, np.int32)
    bbox_max_x = np.full(L, -1, np.int32)
    bbox_min_y = np.full(L, inf, np.int32)
    bbox_max_y = np.full(L, -1, np.int32)

    sum_x = np.zeros(L, np.float32)
    sum_y = np.zeros(L, np.float32)
    count = np.zeros(L, np.int32)

    # 3) One pass over the entire seg array to gather everything
    for i in prange(H):
        for j in range(W):
            lbl = seg[i, j]
            if lbl != 0:
                k = lbl - 1  # 0‐based index
                # Update bbox
                if j < bbox_min_x[k]:
                    bbox_min_x[k] = j
                if j > bbox_max_x[k]:
                    bbox_max_x[k] = j
                if i < bbox_min_y[k]:
                    bbox_min_y[k] = i
                if i > bbox_max_y[k]:
                    bbox_max_y[k] = i

                # Accumulate sum for centroid
                sum_x[k] += j
                sum_y[k] += i
                count[k] += 1

    # 4) Allocate storage for the final outputs
    keypoints = np.zeros((L, 4), np.int32)  # [xmin, xmax, ymin, ymax]
    centerpoints = np.zeros((L, 2), np.float32)  # [xmean, ymean]
    numpixels = np.zeros((L,), np.int32)

    for k in range(L):
        # If a segment label k never appeared, we’ll skip it
        if count[k] == 0:
            # Mark it as invalid by leaving bbox at (0,0,0,0) and centroid at (0,0), numpixels=0
            keypoints[k, 0] = 0
            keypoints[k, 1] = -1
            keypoints[k, 2] = 0
            keypoints[k, 3] = -1
            centerpoints[k, 0] = 0.0
            centerpoints[k, 1] = 0.0
            numpixels[k] = 0
        else:
            keypoints[k, 0] = bbox_min_x[k]
            keypoints[k, 1] = bbox_max_x[k]
            keypoints[k, 2] = bbox_min_y[k]
            keypoints[k, 3] = bbox_max_y[k]
            centerpoints[k, 0] = sum_x[k] / count[k]
            centerpoints[k, 1] = sum_y[k] / count[k]
            numpixels[k] = count[k]

    # 5) Build seg_relabeled:  we want to remap all “old label = s” → “new consecutive label”
    #    Let’s say some labels between 1..L never appeared. We’ll make a map from old→new.
    remap = np.full(L, -1, np.int32)
    new_id = 1
    for k in range(L):
        if count[k] > 0:
            remap[k] = new_id
            new_id += 1
        else:
            remap[k] = 0

    # 6) seg_relabeled: same shape as seg, but every old label s>0 becomes remap[s-1]
    seg_relabeled = np.zeros_like(seg, np.int32)
    for i in prange(H):
        for j in range(W):
            lbl = seg[i, j]
            if lbl != 0:
                seg_relabeled[i, j] = remap[lbl - 1]
            # else stays 0

    # Finally, we only output the stats for those k where count[k] > 0
    valid = numpixels > 0
    num_valid = valid.sum()

    out_keypoints = np.zeros((num_valid, 4), np.int32)
    out_centerpoints = np.zeros((num_valid, 2), np.float32)
    out_numpixels = np.zeros((num_valid,), np.int32)

    idx = 0
    for k in range(L):
        if valid[k]:
            out_keypoints[idx, :] = keypoints[k, :]
            out_centerpoints[idx, :] = centerpoints[k, :]
            out_numpixels[idx] = numpixels[k]
            idx += 1

    return out_keypoints, out_centerpoints, out_numpixels, seg_relabeled


@DATASET_REGISTRY.register()
class KisekiInMemoryInferenceDataset:
    def __init__(self, opt):
        self.data_list = []
        self.samples = []
        self.opt = opt
        self.root = opt["root"]
        self.multi_clip = opt.get("multi_clip", False)
        self.mode = opt.get("mode", "reference")
        if not self.multi_clip:
            character_paths = [self.root]
        else:
            character_paths = [
                osp.join(self.root, character) for character in os.listdir(self.root)
            ]

        GT_REF_COLORIZED_IMGS = {}

        CHARACTER_RAFT_RESOLUTIONS = {}

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
            index_map = {}

            # Get frame numbers for all line_list entries (e.g., [1, 2, ..., 24] for "0001.png", "0002.png", ...)
            line_frame_numbers = [
                self.convert_gt_path_to_int(line_path) for line_path in line_list
            ]
            logger.info(f"Line Frame Numbers: {line_frame_numbers}")

            if self.mode == "forward":
                index_map = {}
                for idx in range(L):  # Iterate over 0-based indices of line_list
                    frame_num = line_frame_numbers[idx]
                    if frame_num not in all_gt:
                        # Ensure the previous index exists (idx > 0)
                        if idx > 0:
                            index_map[idx] = (
                                idx - 1
                            )  # Map current index to previous index
                index_list = list(index_map.keys())
            elif self.mode == "nearest":
                # Adjust to use line_list indices
                index_map = {
                    idx: self._get_ref_frame_id(line_frame_numbers[idx], all_gt)
                    for idx in range(L)
                    if line_frame_numbers[idx] not in all_gt
                }
                index_list = self._sort_indices(index_map)
            elif self.mode == "reference" or self.mode == "end2end":
                index_map = {
                    idx: min(all_gt, key=lambda x: abs(x - frame_num))
                    for idx, frame_num in enumerate(line_frame_numbers)
                    if frame_num not in all_gt
                }
                index_list = list(index_map.keys())
            logger.info(f"Index list: {index_list}")

            for index in index_list:
                file_name, _ = osp.splitext(line_list[index])
                line = line_list[index]
                seg = line.replace("line", "seg")
                ref = None
                # reference mode choose closest frame to the gt frame as reference
                if self.mode == "reference":
                    ref = min(all_gt, key=lambda x: abs(x - line_frame_numbers[index]))
                    file_name_ref, _ = osp.splitext(line_list[ref])
                    line_ref = line_list[ref]
                    seg_ref = line_ref.replace("line", "seg")
                    gt_ref = line_ref.replace("line", "ref") if ref in all_gt else None
                else:
                    ref = index_map[index]
                    file_name_ref, _ = osp.splitext(line_list[ref])
                    line_ref = line_list[ref]
                    seg_ref = line_ref.replace("line", "seg")
                    gt_ref = line_ref.replace("line", "ref") if ref in all_gt else None

                if gt_ref is not None:
                    logger.info(
                        f"GT Ref: {gt_ref}, Ref: {ref}, Index: {index}, Line: {line}, Line Ref: {line_ref} \n"
                    )

                line_file = line

                file_name = file_name
                file_name_ref = file_name_ref

                # read images
                line = read_img_2_np(line)
                line_ref = read_img_2_np(line_ref)

                seg = read_seg_2_np(seg)
                seg_ref = read_seg_2_np(seg_ref)

                gt_ref_key = gt_ref
                gt_ref = read_img_2_np(gt_ref) if gt_ref is not None else None

                line, seg, _ = self._square_img_data(line, seg)
                line_ref, seg_ref, gt_ref = self._square_img_data(
                    line_ref, seg_ref, gt_ref
                )

                keypoints, centerpoints, numpixels, seg = _process_seg(seg)
                keypoints_ref, centerpoints_ref, numpixels_ref, seg_ref = _process_seg(
                    seg_ref
                )

                # np to tensor
                line = torch.from_numpy(line).permute(2, 0, 1) / 255.0
                line_ref = torch.from_numpy(line_ref).permute(2, 0, 1) / 255.0
                seg = torch.from_numpy(seg)[None]
                seg_ref = torch.from_numpy(seg_ref)[None]

                if gt_ref_key not in GT_REF_COLORIZED_IMGS:
                    recolorized_img = (
                        recolorize_seg(seg_ref)
                        if gt_ref is None
                        else recolorize_gt(gt_ref)
                    )
                    GT_REF_COLORIZED_IMGS[gt_ref_key] = recolorized_img
                else:
                    recolorized_img = GT_REF_COLORIZED_IMGS[gt_ref_key]

                if line_root not in CHARACTER_RAFT_RESOLUTIONS:
                    sample_img = cv2.imread(line_file, cv2.IMREAD_UNCHANGED)

                    orig_h, orig_w = sample_img.shape[:2]

                    orig_ratio = orig_w / orig_h
                    raft_resolution = min(
                        RAFT_RESOLUTIONS, key=lambda r: abs((r[0] / r[1]) - orig_ratio)
                    )
                    CHARACTER_RAFT_RESOLUTIONS[line_root] = raft_resolution
                else:
                    raft_resolution = CHARACTER_RAFT_RESOLUTIONS[line_root]

                self.samples.append(
                    {
                        "file_name": file_name,
                        "file_name_ref": file_name_ref,
                        "keypoints": torch.from_numpy(keypoints).unsqueeze(0),
                        "keypoints_ref": torch.from_numpy(keypoints_ref).unsqueeze(0),
                        "centerpoints": torch.from_numpy(centerpoints).unsqueeze(0),
                        "centerpoints_ref": torch.from_numpy(
                            centerpoints_ref
                        ).unsqueeze(0),
                        "numpixels": torch.from_numpy(numpixels).unsqueeze(0),
                        "numpixels_ref": torch.from_numpy(numpixels_ref).unsqueeze(0),
                        "line": line.unsqueeze(0),
                        "line_ref": line_ref.unsqueeze(0),
                        "segment": seg.unsqueeze(0),
                        "segment_ref": seg_ref.unsqueeze(0),
                        "recolorized_img": recolorized_img.unsqueeze(0),
                        "raft_resolution": raft_resolution,
                    }
                )
                """ logger.info(
                    {
                        "file_name": self.samples[-1]["file_name"],
                        "file_name_ref": self.samples[-1]["file_name_ref"],
                        "keypoints": self.samples[-1]["keypoints"].shape,
                        "keypoints_ref": self.samples[-1]["keypoints_ref"].shape,
                        "centerpoints": self.samples[-1]["centerpoints"],
                        "centerpoints_ref": self.samples[-1]["centerpoints_ref"],
                        "numpixels": self.samples[-1]["numpixels"],
                        "numpixels_ref": self.samples[-1]["numpixels_ref"],
                        "line": self.samples[-1]["line"].shape,
                        "line_ref": self.samples[-1]["line_ref"].shape,
                        "segment": self.samples[-1]["segment"].shape,
                        "segment_ref": self.samples[-1]["segment_ref"].shape,
                        "recolorized_img": self.samples[-1]["recolorized_img"].shape,
                    }
                ) """
            logger.info(f"Length of line frames to be colored: {len(self.samples)}")

    def _square_img_data(self, line, seg, gt=None, border=16):
        # Crop the content
        mask = np.any(line != [255, 255, 255], axis=-1)  # assume background is white
        coords = np.argwhere(mask)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        h, w = line.shape[:2]
        y_min, x_min = max(0, y_min - border), max(0, x_min - border)  # Extend border
        y_max, x_max = min(h, y_max + border), min(w, x_max + border)

        line = line[y_min : y_max + 1, x_min : x_max + 1]
        seg = seg[y_min : y_max + 1, x_min : x_max + 1]
        if gt is not None:
            gt = gt[y_min : y_max + 1, x_min : x_max + 1]

        # Pad to square
        nh, nw = line.shape[:2]
        diff = abs(nh - nw)
        pad1, pad2 = diff // 2, diff - diff // 2

        if nh > nw:
            # Width is smaller, pad left and right
            line = np.pad(
                line, ((0, 0), (pad1, pad2), (0, 0)), constant_values=255
            )  # default is 255
            seg = np.pad(
                seg, ((0, 0), (pad1, pad2)), constant_values=0
            )  # 0 will be ignored
            if gt is not None:
                # gt = np.pad(gt, ((0, 0), (pad1, pad2), (0, 0)), mode="edge")
                gt = np.pad(gt, ((0, 0), (pad1, pad2), (0, 0)), constant_values=0)
        else:
            # Height is smaller, pad top and bottom
            line = np.pad(
                line, ((pad1, pad2), (0, 0), (0, 0)), constant_values=255
            )  # default is 255
            seg = np.pad(seg, ((pad1, pad2), (0, 0)), constant_values=0)
            if gt is not None:
                # gt = np.pad(gt, ((pad1, pad2), (0, 0), (0, 0)), mode="edge")
                gt = np.pad(gt, ((pad1, pad2), (0, 0), (0, 0)), constant_values=0)

        return line, seg, gt if gt is not None else None

    """ def _process_seg(self, seg):
        seg_list = np.unique(seg[seg != 0])

        h, w = seg.shape
        hh = np.arange(h)
        ww = np.arange(w)
        xx, yy = np.meshgrid(ww, hh)

        keypoints = []
        centerpoints = []
        numpixels = []
        seg_relabeled = np.zeros_like(seg)

        for i, seg_idx in enumerate(seg_list):
            mask = seg == seg_idx

            xs = xx[mask]
            ys = yy[mask]
            xmin = xs.min()
            xmax = xs.max()
            ymin = ys.min()
            ymax = ys.max()
            xmean = xs.mean()
            ymean = ys.mean()
            keypoints.append([xmin, xmax, ymin, ymax])
            centerpoints.append([xmean, ymean])
            numpixels.append(mask.sum())

            seg_relabeled[mask] = i + 1  # 0 is for black line, start from 1

        keypoints = np.stack(keypoints)
        centerpoints = np.stack(centerpoints)
        numpixels = np.stack(numpixels)

        return keypoints, centerpoints, numpixels, seg_relabeled
 """

    def convert_gt_path_to_int(self, gt_path):
        """Extracts the trailing numeric part from a filename and converts it to an integer."""
        # Extract filename without extension
        filename = osp.splitext(osp.split(gt_path)[-1])[0]

        # Match the LAST sequence of digits at the end of the filename
        match = re.search(r"(\d+)$", filename)  # Ensure digits are at the end

        if match:
            # Convert to integer (automatically drops leading zeros)
            return int(match.group())
        else:
            raise ValueError(f"No trailing numeric part found in filename: {filename}")

    def _get_ref_frame_id(self, index, all_gt):
        nearest_gt = min(all_gt, key=lambda x: abs(x - index))
        ref_index = index - 1 if nearest_gt < index else index + 1
        return ref_index

    def _sort_indices(self, index_map):
        adj_list = defaultdict(list)
        for end, start in index_map.items():
            adj_list[start].append(end)

        visited = set()
        result = []

        def _dfs(point):
            if point not in visited:
                visited.add(point)
                for neighbor in adj_list.get(point, []):
                    _dfs(neighbor)
                result.append(point)

        for point in index_map.keys():
            _dfs(point)

        return result[::-1]


class KisekiParallizedInMemoryInferenceDataset:
    pass
