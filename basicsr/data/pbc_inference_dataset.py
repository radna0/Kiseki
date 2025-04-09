import re
import numpy as np
import os
import os.path as osp
import torch
import torch.utils.data as data
from collections import defaultdict
from glob import glob

from basicsr.utils.registry import DATASET_REGISTRY
from paint.utils import read_img_2_np, read_seg_2_np, recolorize_gt, recolorize_seg
from natsort import natsorted
from numba import njit, prange

import sys

sys.path.append("..")  # Adds higher directory to python modules path.
from utils.logging import logger


class AnimeInferenceDataset:
    def __init__(self):
        self.data_list = []

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

    def _process_seg(self, seg):
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

    def __getitem__(self, index):
        index = index % len(self.data_list)

        file_name = self.data_list[index]["file_name"]
        file_name_ref = self.data_list[index]["file_name_ref"]

        # read images
        line = read_img_2_np(self.data_list[index]["line"])
        line_ref = read_img_2_np(self.data_list[index]["line_ref"])

        seg = read_seg_2_np(self.data_list[index]["seg"])
        seg_ref = read_seg_2_np(self.data_list[index]["seg_ref"])

        gt_ref = self.data_list[index]["gt_ref"]
        gt_ref = read_img_2_np(gt_ref) if gt_ref is not None else None

        line, seg, _ = self._square_img_data(line, seg)
        line_ref, seg_ref, gt_ref = self._square_img_data(line_ref, seg_ref, gt_ref)

        keypoints, centerpoints, numpixels, seg = self._process_seg(seg)
        keypoints_ref, centerpoints_ref, numpixels_ref, seg_ref = self._process_seg(
            seg_ref
        )

        # np to tensor
        line = torch.from_numpy(line).permute(2, 0, 1) / 255.0
        line_ref = torch.from_numpy(line_ref).permute(2, 0, 1) / 255.0
        seg = torch.from_numpy(seg)[None]
        seg_ref = torch.from_numpy(seg_ref)[None]

        recolorized_img = (
            recolorize_seg(seg_ref) if gt_ref is None else recolorize_gt(gt_ref)
        )

        return {
            "file_name": file_name,
            "file_name_ref": file_name_ref,
            "keypoints": keypoints,
            "keypoints_ref": keypoints_ref,
            "centerpoints": centerpoints,
            "centerpoints_ref": centerpoints_ref,
            "numpixels": numpixels,
            "numpixels_ref": numpixels_ref,
            "line": line,
            "line_ref": line_ref,
            "segment": seg,
            "segment_ref": seg_ref,
            "recolorized_img": recolorized_img,
        }

    def __rmul__(self, v):
        self.data_list = v * self.data_list
        return self

    def __len__(self):
        return len(self.data_list)


@DATASET_REGISTRY.register()
class PaintBucketInferenceDataset(AnimeInferenceDataset):
    def __init__(self, opt):
        super(PaintBucketInferenceDataset, self).__init__()

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
            elif self.mode == "reference":
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

                data_sample = {
                    "file_name": file_name,
                    "line": line,
                    "seg": seg,
                    "file_name_ref": file_name_ref,
                    "line_ref": line_ref,
                    "seg_ref": seg_ref,
                    "gt_ref": gt_ref,
                }
                self.data_list += [data_sample]

        logger.info(f"Length of line frames to be colored: {len(self.data_list)}")

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


""" 
import re
import numpy as np
import os
import os.path as osp
import torch
import torch.utils.data as data
from collections import defaultdict
from glob import glob

from basicsr.utils.registry import DATASET_REGISTRY
from paint.utils import read_img_2_np, read_seg_2_np, recolorize_gt, recolorize_seg
from natsort import natsorted

from numba import njit, prange


class AnimeInferenceDataset:
    def __init__(self):
        self.preloaded_data = []

    def _process_seg(self, seg):
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
            line = np.pad(line, ((0, 0), (pad1, pad2), (0, 0)), constant_values=255) #default is 255
            seg = np.pad(seg, ((0, 0), (pad1, pad2)), constant_values=0)  # 0 will be ignored
            if gt is not None:
                #gt = np.pad(gt, ((0, 0), (pad1, pad2), (0, 0)), mode="edge")
                gt = np.pad(gt, ((0, 0), (pad1, pad2), (0, 0)), constant_values=0)
        else:
            # Height is smaller, pad top and bottom
            line = np.pad(line, ((pad1, pad2), (0, 0), (0, 0)), constant_values=255) #default is 255
            seg = np.pad(seg, ((pad1, pad2), (0, 0)), constant_values=0)
            if gt is not None:
                #gt = np.pad(gt, ((pad1, pad2), (0, 0), (0, 0)), mode="edge")
                gt = np.pad(gt, ((pad1, pad2), (0, 0), (0, 0)), constant_values=0)

        return line, seg, gt if gt is not None else None

    def load_single(self, data_info):
            # Load and process single sample
            line = read_img_2_np(data_info["line"]).astype(np.uint8)
            line_ref = read_img_2_np(data_info["line_ref"]).astype(np.uint8)
            seg = read_seg_2_np(data_info["seg"])
            seg_ref = read_seg_2_np(data_info["seg_ref"])
            gt_ref = read_img_2_np(data_info["gt_ref"]) if data_info["gt_ref"] else None

            # Process with Numba optimized function
            line_pad, seg_pad, _ = self._square_img_data(line, seg, None)
            line_ref_pad, seg_ref_pad, gt_ref_pad = self._square_img_data(line_ref, seg_ref, gt_ref)

            # Process segmentation
            keypoints, centerpoints, numpixels, seg = self._process_seg(seg_pad)
            keypoints_ref, centerpoints_ref, numpixels_ref, seg_ref = self._process_seg(seg_ref_pad)

            # Convert to tensors
            return {
                "file_name": data_info["file_name"],
                "file_name_ref": data_info["file_name_ref"],
                "line": torch.from_numpy(line_pad).permute(2, 0, 1).float() / 255.0,
                "line_ref": torch.from_numpy(line_ref_pad).permute(2, 0, 1).float() / 255.0,
                "segment": torch.from_numpy(seg).unsqueeze(0),
                "segment_ref": torch.from_numpy(seg_ref).unsqueeze(0),
                "keypoints": torch.from_numpy(keypoints),
                "keypoints_ref": torch.from_numpy(keypoints_ref),
                "centerpoints": torch.from_numpy(centerpoints),
                "centerpoints_ref": torch.from_numpy(centerpoints_ref),
                "numpixels": torch.from_numpy(numpixels),
                "numpixels_ref": torch.from_numpy(numpixels_ref),
                "recolorized_img": recolorize_gt(gt_ref_pad) if gt_ref_pad is not None 
                                else recolorize_seg(seg_ref)
            }


@DATASET_REGISTRY.register()
class PaintBucketInferenceDataset(AnimeInferenceDataset):
    def __init__(self, opt):
        super().__init__()
        self.data_list = []
        self.opt = opt
        self.root = opt["root"]
        self.multi_clip = opt.get("multi_clip", False)
        self.mode = opt.get("mode", "forward")
        self._build_data_list()
        self._preload_all()

    def _build_data_list(self):
        character_paths = [self.root] if not self.multi_clip else [
            osp.join(self.root, c) for c in os.listdir(self.root)
        ]

        for char_path in character_paths:
            line_root = osp.join(char_path, "line")
            gt_root = osp.join(char_path, "ref")
            
            line_files = natsorted(glob(osp.join(line_root, "*.png")))
            gt_files = {self._path_to_frame(p): p for p in glob(osp.join(gt_root, "*.png"))}
            
            line_frames = {self._path_to_frame(p): p for p in line_files}
            all_gt_frames = set(gt_files.keys())
            
            index_map = self._create_index_map(line_frames, all_gt_frames)
            self._populate_data_list(char_path, line_files, index_map, gt_files)

    def _preload_all(self):
        self.preloaded_data = [self.load_single(item) for item in self.data_list]

    def _path_to_frame(self, path):
        filename = osp.splitext(osp.basename(path))[0]
        match = re.search(r'\d+$', filename)
        return int(match.group()) if match else 0

    def _create_index_map(self, line_frames, gt_frames):
        frames = sorted(line_frames.keys())
        gt_set = set(gt_frames)
        index_map = {}

        for idx, frame in enumerate(frames):
            if frame not in gt_set:
                if self.mode == "forward" and idx > 0:
                    index_map[idx] = idx - 1
                elif self.mode == "nearest":
                    nearest = min(gt_frames, key=lambda x: abs(x - frame))
                    index_map[idx] = frames.index(nearest)
                    
        return index_map

    def _populate_data_list(self, char_path, line_files, index_map, gt_files):
        for idx, ref_idx in index_map.items():
            line_path = line_files[idx]
            ref_path = line_files[ref_idx]
            
            self.data_list.append({
                "file_name": osp.splitext(osp.basename(line_path))[0],
                "line": line_path,
                "seg": line_path.replace("line", "seg"),
                "file_name_ref": osp.splitext(osp.basename(ref_path))[0],
                "line_ref": ref_path,
                "seg_ref": ref_path.replace("line", "seg"),
                "gt_ref": gt_files.get(self._path_to_frame(ref_path), None)
            })

    def __len__(self):
        return len(self.preloaded_data)

    def __getitem__(self, index):
        return self.preloaded_data[index] """
