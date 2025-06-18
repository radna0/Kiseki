import numpy as np
import cv2
from linefiller.linefiller.trappedball_fill import (
    trapped_ball_fill_multi,
    flood_fill_multi,
    mark_fill,
    build_fill_map,
    merge_fill,
    show_fill_map,
)
from linefiller.linefiller.thinning import thinning
import time
import argparse
from PIL import Image
from kiseki.logging import logger


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


def processing(img_path, radius=4, contour=False) -> np.ndarray:
    # trappedball_fill_numba(img_path, save_path, radius, contour)
    # return

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
    return fillmap


def saveAll(fillmap: np.ndarray, PATH: str) -> None:
    # color+undertone
    cv2.imwrite(PATH + "fills_merged_no_contour.png", show_fill_map(thinning(fillmap)))
    # undertone
    cv2.imwrite(PATH + "fills_merged.png", show_fill_map(fillmap))


def main() -> None:
    parser = argparse.ArgumentParser(description="Line Filler")
    # args
    parser.add_argument(
        "-im", "--image", type=str, help="Image Path", default="example.png"
    )
    parser.add_argument("-o", "--output", type=str, help="Save Root Path", default="./")

    args = parser.parse_args()

    logger.info("Start!")
    start = time.time()
    fillmap = processing(img_path=args.image)
    saveAll(fillmap=fillmap, PATH=args.output)
    logger.info("All Finished...")
    logger.info(f"Total Running Time : {time.time() - start :.2f}sec")


if __name__ == "__main__":
    main()
