import os
import shutil
import cv2
import xxhash
import imagehash
from PIL import Image
from natsort import natsorted
from skimage.metrics import structural_similarity as ssim
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import os.path as osp
from glob import glob

from ..logging import logger

exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
cv2.setUseOptimized(True)
cv2.ocl.setUseOpenCL(True)


def get_image_signature(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    xxh = xxhash.xxh64()
    xxh.update(img.tobytes())
    resized = cv2.resize(img, (32, 32), interpolation=cv2.INTER_LANCZOS4)
    gray = (
        cv2.cvtColor(resized, cv2.COLOR_BGRA2GRAY)
        if resized.shape[2] == 4
        else cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    )
    phash = imagehash.dhash(Image.fromarray(gray), hash_size=32)
    ssim_img = cv2.resize(gray, (1024, 1024), interpolation=cv2.INTER_AREA)
    return {
        "path": image_path,
        "xxhash": xxh.hexdigest(),
        "phash": phash,
        "ssim_img": ssim_img,
        "original": None,
    }


def preprocess_parallel(files):
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(get_image_signature, f) for f in files]
        results = [f.result() for f in as_completed(futures) if f.result()]
    return natsorted(results, key=lambda x: x["path"])


def analyze_sequence(folder, ssim_threshold=0.995):
    files = natsorted(
        [
            osp.join(folder, f)
            for f in os.listdir(folder)
            if osp.splitext(f)[1].lower() in exts
        ]
    )
    logger.info(f"Processing {len(files)} frames for duplicates…")
    sigs = preprocess_parallel(files)

    xxmap, phmap, dup_counts = {}, {}, {}
    for sig in sigs:
        # exact‐hash duplicates
        if sig["xxhash"] in xxmap:
            orig = xxmap[sig["xxhash"]]
            sig["original"] = orig
            dup_counts[orig] = dup_counts.get(orig, 0) + 1
            continue
        xxmap[sig["xxhash"]] = sig["path"]

        # perceptual‐hash + SSIM
        matched = False
        for ehash, esig in phmap.items():
            if (sig["phash"] - ehash) < 3:
                sim = ssim(
                    sig["ssim_img"],
                    esig["ssim_img"],
                    data_range=esig["ssim_img"].max() - esig["ssim_img"].min(),
                )
                if sim >= ssim_threshold:
                    sig["original"] = esig["path"]
                    dup_counts[esig["path"]] = dup_counts.get(esig["path"], 0) + 1
                    matched = True
                    break
        if not matched:
            phmap[sig["phash"]] = sig

    key_frames = []
    for sig in sigs:
        if sig["original"] is None:
            exact = sum(
                1
                for s in sigs
                if s.get("original") == sig["path"] and s["xxhash"] == sig["xxhash"]
            )
            key_frames.append({"path": sig["path"], "exact_duplicates": exact})
    return natsorted(key_frames, key=lambda x: x["path"])


def main(path):
    # Locate raw folders
    raw_ref = osp.join(path, "ref_raw")
    raw_line = osp.join(path, "line_raw")
    if not osp.isdir(raw_ref) or not osp.isdir(raw_line):
        logger.error("Need both 'ref_raw/' and 'line_raw/' under the given path.")
        return

    # Prepare processing folders
    ref_dir = osp.join(path, "ref")
    line_dir = osp.join(path, "line")
    for d in (ref_dir, line_dir):
        if osp.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    # 1) Copy all refs with simple rename‐map
    raw_ref_map = {}
    refs = natsorted(
        [f for f in os.listdir(raw_ref) if osp.splitext(f)[1].lower() in exts]
    )
    for i, orig in enumerate(refs):
        ext = osp.splitext(orig)[1]
        new = f"{i}{ext}"
        shutil.copy(osp.join(raw_ref, orig), osp.join(ref_dir, new))
        raw_ref_map[new] = orig
    logger.info(f"Raw Ref Map: {raw_ref_map}")

    # 2) Copy all lines raw → line_dir
    lines = natsorted(
        [f for f in os.listdir(raw_line) if osp.splitext(f)[1].lower() in exts]
    )
    for orig in lines:
        shutil.copy(osp.join(raw_line, orig), osp.join(line_dir, orig))

    # 3) Deduplicate in-place under line_dir
    key_frames = analyze_sequence(line_dir)

    raw_line_map = {}
    dups_sorted = []
    tmp_file_names = []
    keep = set()

    # 4) Copy & rename key‐frames into line_dir
    for i, kf in enumerate(key_frames):
        orig_basename = osp.basename(kf["path"])
        ext = osp.splitext(orig_basename)[1]
        new_name = f"{i}{ext}"
        shutil.copy(kf["path"], osp.join(line_dir, new_name))
        raw_line_map[new_name] = orig_basename
        dups_sorted.append(kf["exact_duplicates"])
        tmp_file_names.append(new_name)
        keep.add(new_name)

    # 5) Remove any leftover files in line_dir not in keep
    for f in os.listdir(line_dir):
        if f not in keep and osp.splitext(f)[1].lower() in exts:
            os.remove(osp.join(line_dir, f))

    logger.info(f"Raw Line Map: {raw_line_map}\n")
    logger.info(f"key_frames: {dups_sorted}\n")
    logger.info(f"tmp_file_names: {tmp_file_names}\n")

    return raw_ref_map, dups_sorted, tmp_file_names, raw_line_map


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy from ref_raw/ & line_raw/, then dedupe line/ in-place"
    )
    parser.add_argument("--path", default=".")
    args = parser.parse_args()

    results = main(args.path)
    print("Results:", results)
