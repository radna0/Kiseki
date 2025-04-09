import os
import shutil
import cv2
import xxhash
import imagehash
import numpy as np
from PIL import Image
from natsort import natsorted
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import os.path as osp
from glob import glob

from .logging import logger

# Enable hardware acceleration
cv2.setUseOptimized(True)
cv2.ocl.setUseOpenCL(True)


# --------------------------
# Core Optimization Functions
# --------------------------
def get_image_signature(image_path):
    """Compute all required signatures in one pass"""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None

    # Fast hashing
    xxh = xxhash.xxh64()
    xxh.update(img.tobytes())

    # Perceptual hash
    resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LANCZOS4)
    gray = (
        cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY)
        if img.shape[2] == 4
        else cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    )
    phash = imagehash.dhash(Image.fromarray(gray), hash_size=32)

    # SSIM pre-processing
    ssim_img = cv2.resize(gray, (1024, 1024), interpolation=cv2.INTER_AREA)

    return {
        "path": image_path,
        "xxhash": xxh.hexdigest(),
        "phash": phash,
        "ssim_img": ssim_img,
        "original": None,  # Will track original frame later
    }


def preprocess_parallel(image_files):
    """Parallel processing with natural order preservation"""
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(get_image_signature, f) for f in image_files]
        return natsorted(
            [f.result() for f in as_completed(futures)], key=lambda x: x["path"]
        )


# --------------------------
# Main Processing
# --------------------------
def analyze_sequence(image_folder, ssim_threshold=0.995):
    # Get and sort image files naturally
    image_files = natsorted(
        [
            os.path.join(image_folder, f)
            for f in os.listdir(image_folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
        ]
    )

    # Parallel preprocessing
    logger.info(f"Processing images: {len(image_files)} files")
    signatures = preprocess_parallel(image_files)

    # Track originals and duplicates
    xxhash_map = {}
    phash_map = {}
    duplicate_counts = {}

    # Process in natural order to ensure first occurrence priority
    for sig in signatures:
        if sig is None:
            continue

        # Exact duplicates check
        if sig["xxhash"] in xxhash_map:
            original = xxhash_map[sig["xxhash"]]
            sig["original"] = original
            duplicate_counts[original] = duplicate_counts.get(original, 0) + 1
            continue
        else:
            xxhash_map[sig["xxhash"]] = sig["path"]

        # Near-duplicates check
        match_found = False
        for existing_phash in list(phash_map.keys()):
            if (sig["phash"] - existing_phash) < 3:
                existing_sig = phash_map[existing_phash]
                similarity = ssim(
                    sig["ssim_img"],
                    existing_sig["ssim_img"],
                    data_range=existing_sig["ssim_img"].max()
                    - existing_sig["ssim_img"].min(),
                )

                if similarity >= ssim_threshold:
                    sig["original"] = existing_sig["path"]
                    duplicate_counts[existing_sig["path"]] = (
                        duplicate_counts.get(existing_sig["path"], 0) + 1
                    )
                    match_found = True
                    break

        if not match_found:
            phash_map[sig["phash"]] = sig

    # Generate final report
    key_frames = []
    for sig in signatures:
        if sig["original"] is None:
            key_frames.append(
                {
                    "path": sig["path"],
                    "total_duplicates": duplicate_counts.get(sig["path"], 0),
                    "exact_duplicates": 0,
                    "near_duplicates": 0,
                }
            )

    # Count exact vs near duplicates
    for sig in signatures:
        if sig["original"] is not None:
            original_path = sig["original"]
            for kf in key_frames:
                if kf["path"] == original_path:
                    if sig["phash"] == phash_map[sig["phash"]]["phash"]:
                        kf["exact_duplicates"] += 1
                    else:
                        kf["near_duplicates"] += 1
                    break

    return natsorted(key_frames, key=lambda x: x["path"])


# --------------------------
# Usage
# --------------------------


def get_file_name(file_path):
    return osp.split(file_path)[-1]


def get_path_from_raw_path(index, raw):
    return f'{index}.{raw.split(".")[-1]}'


def copy_raw_to_process(raw_folder, process_folder):
    os.makedirs(process_folder, exist_ok=True)
    raw_list = natsorted(glob(osp.join(raw_folder, "*.png")))
    # sort numerically
    raw_to_process = {
        get_path_from_raw_path(i, raw): get_file_name(raw)
        for i, raw in enumerate(raw_list)
    }

    # based on the ref_raw, copy the ref_raw to ref folder
    for process_file, raw_file in raw_to_process.items():
        shutil.copy(
            osp.join(raw_folder, raw_file), osp.join(process_folder, process_file)
        )

    return raw_to_process


def process_key_frames(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    results = analyze_sequence(src_folder)

    logger.info(f"Key frames with duplicate counts: Total: {len(results)} files")
    for kf in results:
        filename = os.path.basename(kf["path"])
        # logger.info(f" {filename} - Exact duplicates: {kf['exact_duplicates']} \n")
        # save the key frames to the dst_folder folder
        shutil.copy(kf["path"], dst_folder)

    return results


def main(path):
    ref_raw = osp.join(path, "ref_raw")
    line_raw = osp.join(path, "line_raw")
    if not osp.exists(ref_raw) and not osp.exists(line_raw):
        logger.info("ref_raw and line_raw not found.")
        return

    line_raw_uniq = osp.join(path, "line_raw_uniq")
    if osp.exists(line_raw_uniq):
        shutil.rmtree(line_raw_uniq)
    key_frames = process_key_frames(line_raw, line_raw_uniq)

    ref = osp.join(path, "ref")
    line = osp.join(path, "line")
    if osp.exists(ref) and osp.exists(line):
        shutil.rmtree(ref)
        shutil.rmtree(line)

    raw_ref_map = copy_raw_to_process(ref_raw, ref)
    logger.info(f"Raw Ref Map: {raw_ref_map}\n")
    raw_line_map = copy_raw_to_process(line_raw_uniq, line)
    logger.info(f"Raw Line Map: {raw_line_map}\n")

    line_list_sorted = natsorted(glob(osp.join(line, "*.png")))
    tmp_file_names = [osp.split(x)[-1] for x in line_list_sorted]

    dups_sorted = [res["exact_duplicates"] for res in key_frames]
    logger.info(f"key_frames: {dups_sorted}\n")
    logger.info(f"tmp_file_names: {tmp_file_names}\n")
    return dups_sorted, tmp_file_names


if __name__ == "__main__":
    import time

    start = time.time()
    parser = argparse.ArgumentParser(
        description="Find key frames in a sequence of images"
    )
    parser.add_argument(
        "--folder",
        help="Path to folder containing PNG files",
        default="dataset/test/manu/line_raw",
    )
    args = parser.parse_args()
    results = analyze_sequence(args.folder)

    print("\nKey frames with duplicate counts:")
    for kf in results:
        filename = os.path.basename(kf["path"])
        print(f" {filename} - Exact duplicates: {kf['exact_duplicates']} \n")

    end = time.time()
    print(f"Processing time: {end - start:.2f} seconds")
