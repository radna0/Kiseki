import os
import cv2
import shutil
from glob import glob
from os import path as osp
import multiprocessing
from .logging import logger, Profiler

from linefiller.linefiller.thinning import thinning
from linefiller.linefiller.trappedball_fill_opti import (
    build_fill_map,
    flood_fill_multi,
    mark_fill,
    merge_fill,
    show_fill_map,
    trapped_ball_fill_multi,
)
from PIL import Image
import numpy as np
from skimage import color, io, measure, morphology
from skimage.measure import label
from skimage.morphology import binary_dilation, square, footprint_rectangle
from tqdm import tqdm

from paint.lineart import LineArt

from numba import njit, prange
from paint.utils import (
    dump_json,
    np_2_labelpng,
    process_gt,
    read_line_2_np,
    read_seg_2_np,
    recolorize_seg,
    colorize_label_image,
)


def extract_color_dict(gt_path, seg_path):
    gt = io.imread(gt_path)
    seg = read_seg_2_np(seg_path)
    gt = process_gt(gt, seg)
    color_dict = {}
    props = measure.regionprops(seg)
    for i in range(1, seg.max() + 1):
        pos = props[i - 1].coords[0]
        index_color = gt[pos[0], pos[1], :]
        color_dict[str(i)] = index_color.tolist()
    save_path = seg_path.replace(".png", ".json")
    dump_json(color_dict, save_path)


@njit(nogil=True)
def compute_top_left_coords(label_img, max_label):
    """
    For each label 1..max_label, compute the minimum (row, col) where it occurs.
    For labels that do not occur the values remain INF.
    """
    h, w = label_img.shape
    INF = 10**9  # a very large number
    # Preallocate arrays: index 0 is unused (or for the line art)
    top_rows = np.full(max_label + 1, INF, dtype=np.int32)
    top_cols = np.full(max_label + 1, INF, dtype=np.int32)

    for i in range(h):
        for j in range(w):
            lab = label_img[i, j]
            if lab != 0:
                # update row coordinate if needed
                if i < top_rows[lab]:
                    top_rows[lab] = i
                    top_cols[lab] = j  # also update col since this row is better
                # if same row, update col if needed
                elif i == top_rows[lab] and j < top_cols[lab]:
                    top_cols[lab] = j
    return top_rows, top_cols


def relabel_image(label_img, color_dict):
    """
    Relabel the image by reassigning new labels in order of the top-left pixel
    (i.e. smallest row, and then smallest column if needed).  Also, update the color dictionary.
    """
    # Get the set of unique labels present.
    unique_labels = np.unique(label_img)
    relabeled_img = np.zeros_like(label_img)
    recolored_dict = {}

    # Compute maximum label (assumes labels are nonnegative integers)
    max_label = int(unique_labels.max())

    # Compute top-left coordinates for each label (using a fast, compiled loop)
    top_rows, top_cols = compute_top_left_coords(label_img, max_label)

    # We are not interested in label 0 (the background / line art)
    # Gather labels that occur (i.e. whose top_rows are not INF)
    valid_labels = unique_labels[unique_labels != 0]

    # For these valid labels, get their top-left coordinates.
    # (If multiple labels share the same top row, we break ties by column.)
    valid_top_rows = top_rows[valid_labels]
    valid_top_cols = top_cols[valid_labels]

    # Sort labels in lexicographical order (first by row then by col)
    sort_indices = np.lexsort((valid_top_cols, valid_top_rows))
    sorted_labels = valid_labels[sort_indices]

    # Now reassign new labels in the sorted order.
    new_label = 1
    for lab in sorted_labels:
        # Create a mask for the current original label.
        mask = label_img == lab
        relabeled_img[mask] = new_label

        # Update the recolored dictionary. (Assumes that color_dict keys are strings.)
        recolored_dict[str(new_label)] = color_dict[str(lab)]
        new_label += 1

    return relabeled_img, recolored_dict


def extract_label_map(
    color_img_path, img_save_path=None, line_img_path=None, extract_seg=False
):
    # This part is mainly for the test data.
    # This function will extract the label map for each colorized img.
    # Every connected segment will be viewed as one label.
    # If the extract_seg is true, this module will extract the segment rather than label
    img = io.imread(color_img_path)
    img = np.array(img)
    if line_img_path is not None:
        line = np.array(io.imread(line_img_path))
    labeled_img = np.zeros(img.shape[:2], dtype=np.int32)
    color_dict = {}
    neighborhood = footprint_rectangle((3, 3))
    index = 0  # index 0 means the black line.

    img_data = img[:, :, :3]
    colors = img_data.reshape(-1, img_data.shape[-1])
    color_list = list(np.unique(colors, axis=0))
    # Background is the most bright
    for i, color in enumerate(color_list):
        mask = np.all(img[:, :, :3] == color, axis=-1)
        if np.max(mask) != 0 and np.max(color) != 0:
            if not extract_seg:
                expanded_mask = binary_dilation(mask, footprint=neighborhood)
            else:
                expanded_mask = mask
            labeled_color_regions, num_labels = label(
                expanded_mask, connectivity=2, return_num=True
            )
            labeled_color_regions = labeled_color_regions * mask
            for region_label in range(1, num_labels + 1):
                region_mask = labeled_color_regions == region_label
                if np.sum(region_mask) > 1:
                    # We do not calculate accuracy for single pixel region.
                    index += 1
                    labeled_img[region_mask] = index

                    if np.min(color) == 255:
                        color_new = np.append(color, 0)
                    else:
                        color_new = np.append(color, 255)
                    color_dict[str(index)] = color_new.tolist()
                else:
                    # fill the line art and gt's single pixel
                    if line_img_path is not None:
                        line[region_mask] = [0, 0, 0, 255]
                    img[region_mask] = [0, 0, 0, 255]
    # relabel all the given labels to avoid data leakage
    labeled_img, color_dict = relabel_image(labeled_img, color_dict)

    if img_save_path is not None:
        np_2_labelpng(labeled_img, img_save_path)
        # dump_json(color_dict, img_save_path.replace(".png", ".json"))
    if line_img_path is not None:
        io.imsave(line_img_path, line, check_contrast=False)
        io.imsave(color_img_path, img, check_contrast=False)
    return labeled_img


def read_line_2_np(img_path, channel=4):
    img = Image.open(img_path)
    img_np = np.array(img)

    if img.mode == "RGBA":
        alpha_channel = img_np[:, :, 3]
        mask = alpha_channel > 100  # Line detection based on alpha value, default is 10
    elif img.mode == "RGB":
        grayscale = np.mean(img_np[:, :, :3], axis=2)
        mask = (
            grayscale < 150
        )  # Line detection based on grayscale value, default is 245

    line = np.zeros((*img_np.shape[:2], 4), dtype=np.uint8)
    line[:, :, :3] = 255  # Set all RGB to white
    line[:, :, 3] = np.where(mask, 255, 0)  # Set alpha: 255 for lines, 0 for background

    # Copy original RGB values to new image where there are lines
    line[mask, :3] = img_np[mask, :3]

    return line[..., :channel]


def trappedball_fill(img_path, save_path, radius=4, contour=False):

    im = read_line_2_np(img_path, channel=3)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(im, 220, 255, cv2.THRESH_BINARY)

    fills = []
    result = binary.copy()

    # Precompute all required radii
    radii = [min(20, radius), min(10, radius // 2), 1]

    # Batch process fill operations
    for r, method in zip(radii, ["max", "mean", "mean"]):
        fill = trapped_ball_fill_multi(result, r, method)
        fills.extend(fill)
        result = mark_fill(result, fill)  # In-place modification

    fill = flood_fill_multi(result)
    fills.extend(fill)

    fillmap = build_fill_map(result, fills)
    fillmap = merge_fill(fillmap)

    cv2.imwrite(save_path, show_fill_map(thinning(fillmap) if not contour else fillmap))


def extract_seg_from_color(color_img_path, line_path, seg_save_path):
    extract_label_map(color_img_path, seg_save_path, line_path, extract_seg=True)


def process_line(
    line_path, seg_folder, seg_color_folder, radius, ref_folder, ref_names
):
    """Process a line path.

    Args:
        line_path (str): Path to the line path.
        seg_folder (str): Path to the segmentation folder.
        seg_color_folder (str): Path to the segmentation color folder.
        radius (int): Radius of the line.
        ref_folder (str): Path to the reference folder.
        ref_names (list[str]): List of reference names.
    """
    name = osp.split(line_path)[-1][:-4]
    seg_path = osp.join(seg_folder, name + ".png")
    seg_color_path = osp.join(seg_color_folder, name + ".png")

    # profile trappedball_fill
    with Profiler("TrappedBall Filling Time", limit=0):
        trappedball_fill(line_path, seg_color_path, radius, contour=True)

    # return

    with Profiler("Extracting Segments Time", limit=0):
        extract_seg_from_color(seg_color_path, line_path, seg_path)

    with Profiler("Colorizing Segments Time", limit=0):
        if name in ref_names:
            ref_path = osp.join(ref_folder, name + ".png")
            extract_color_dict(ref_path, seg_path)
            colorize_label_image(seg_path, seg_path.replace(".png", ".json"), ref_path)

    logger.info(f"{seg_path} created.")


def main(args):
    if not args.multi_clip:
        clip_list = [args.path]
    else:
        clip_list = [osp.join(args.path, clip) for clip in os.listdir(args.path)]

    for clip_path in clip_list:
        ref_folder = osp.join(clip_path, "ref")
        line_folder = osp.join(clip_path, "line")
        seg_folder = osp.join(clip_path, "seg")
        seg_color_folder = osp.join(clip_path, "seg_color")

        ref_names = [osp.splitext(ref)[0] for ref in os.listdir(ref_folder)]
        os.makedirs(seg_folder, exist_ok=True)
        os.makedirs(seg_color_folder, exist_ok=True)

        # Prepare arguments for multiprocessing
        line_paths = sorted(glob(osp.join(line_folder, "*.png")))
        process_args = [
            (
                line_path,
                seg_folder,
                seg_color_folder,
                args.radius,
                ref_folder,
                ref_names,
            )
            for line_path in line_paths
        ]

        # Use multiprocessing Pool to process line paths in parallel
        with multiprocessing.Pool() as pool:
            pool.starmap(process_line, process_args)
