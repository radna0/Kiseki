#!/usr/bin/env python3
import os
import time
from PIL import Image
import numpy as np
import cv2
from os import path as osp


def is_lineart_image(image_path):
    """
    Checks if image is already a pure black-on-transparent line-art PNG:
    - Mode must be RGBA
    - All pixels with alpha>0 must have RGB values == 0 (black lines)
    - Pixels with alpha==0 are background
    """
    try:
        with Image.open(image_path) as img:
            if img.mode != "RGBA":
                return False
            arr = np.array(img)
    except Exception:
        return False

    alpha = arr[:, :, 3]
    rgb = arr[:, :, :3]
    # mask of drawn pixels
    mask = alpha > 0
    # on mask pixels, rgb must be black
    if not np.all(rgb[mask] == 0):
        return False
    # ensure no semi-transparent lines? optional: enforce alpha==255
    if not np.all(alpha[mask] == 255):
        return False
    return True


def detect_line_art(
    image,
    adaptive_block_size=15,
    adaptive_C=2,
    bilateral_d=9,
    bilateral_sigmaColor=75,
    bilateral_sigmaSpace=75,
    open_kernel_size=3,
):
    """
    Robust line-art detection combining bilateral filtering, adaptive thresholding,
    morphological opening, and optional thinning.

    Returns a 2D boolean mask where True indicates line pixels.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.array(image)
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    filtered = cv2.bilateralFilter(
        gray,
        d=bilateral_d,
        sigmaColor=bilateral_sigmaColor,
        sigmaSpace=bilateral_sigmaSpace,
    )
    thresh = cv2.adaptiveThreshold(
        filtered,
        maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=cv2.THRESH_BINARY_INV,
        blockSize=adaptive_block_size,
        C=adaptive_C,
    )
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (open_kernel_size, open_kernel_size)
    )
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    if hasattr(cv2.ximgproc, "thinning"):
        mask = cv2.ximgproc.thinning(opened) > 0
    else:
        mask = opened > 0
    return mask


def process_directory(input_dir, output_dir, **kwargs):
    """Process all images, generating transparent line-art PNGs with only black lines."""
    os.makedirs(output_dir, exist_ok=True)
    start_time = time.time()
    processed_any = False

    for fname in sorted(os.listdir(input_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
            continue
        in_path = osp.join(input_dir, fname)
        base, _ = osp.splitext(fname)
        out_path = osp.join(output_dir, f"{base}.png")

        # Skip if already proper line-art
        if is_lineart_image(in_path):
            # Simply copy to output
            with Image.open(in_path).convert("RGBA") as img:
                img.save(out_path)
            continue

        # Otherwise detect lines
        with Image.open(in_path) as img:
            mask = detect_line_art(img, **kwargs)
            processed_any = True

        # Build RGBA array: black lines (mask), transparent bg
        h, w = mask.shape
        line_np = np.zeros((h, w, 4), dtype=np.uint8)
        line_np[:, :, 3] = mask.astype(np.uint8) * 255
        out_img = Image.fromarray(line_np, mode="RGBA")
        out_img.save(out_path)

    duration = time.time() - start_time
    print(
        f"[Step 1.5] Robust line-art processing completed in {duration:.2f}s. "
        f"Processed any: {processed_any}. Output dir: {output_dir}"
    )
    return processed_any, output_dir, duration


def main(args):
    input_dir = osp.join(args.path, "line")
    output_dir = osp.join(args.path, "line_processed")
    process_directory(
        input_dir,
        output_dir,
        adaptive_block_size=getattr(args, "adaptive_block_size", 15),
        adaptive_C=getattr(args, "adaptive_C", 2),
        bilateral_d=getattr(args, "bilateral_d", 9),
        bilateral_sigmaColor=getattr(args, "bilateral_sigmaColor", 75),
        bilateral_sigmaSpace=getattr(args, "bilateral_sigmaSpace", 75),
        open_kernel_size=getattr(args, "open_kernel_size", 3),
    )
