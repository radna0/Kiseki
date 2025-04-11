import argparse
import os
from PIL import Image
from natsort import natsorted
import os.path as osp
from glob import glob
import shutil
import sys

sys.path.append("..")  # Adds higher directory to python modules path.

from utils.logging import logger


def create_pngs(path, dups_sorted, tmp_file_names, final_path):
    # path: dataset/test/manu_test/, path_main_dir: manu_test
    path_main_dir = os.path.basename(os.path.normpath(path))
    line_raw = osp.join(path, "line_raw")
    raw_list = natsorted(glob(osp.join(line_raw, "*.png")))
    raw_list_names = [osp.split(x)[-1] for x in raw_list]
    logger.info(f"Sorted Line Raw List: {raw_list_names}\n")
    final_dir = os.path.join(path, final_path)
    os.makedirs(final_dir, exist_ok=True)

    output_idx = 0  # to name the new files sequentially

    for file_name, num_dups in zip(tmp_file_names, dups_sorted):
        logger.info(
            f"Processing {file_name} with {num_dups} duplicates. path: {path}, path_main_dir: {path_main_dir}, file_name: {file_name}"
        )
        src_path = os.path.join(path, path_main_dir, file_name)
        for _ in range(num_dups + 1):  # include the original + duplicates
            dst_name = raw_list_names[output_idx]
            dst_path = os.path.join(final_dir, dst_name)
            shutil.copy(src_path, dst_path)
            output_idx += 1


def create_gif(folder_path, output_path="output.gif", duration=None):
    # Get all PNG files in the folder
    files = os.listdir(folder_path)
    png_files = [f for f in files if f.lower().endswith(".png")]

    # Sort files numerically based on their names (ignoring extension)
    sorted_files = natsorted(
        [(_, filename) for _, filename in enumerate(png_files)],
        key=lambda x: x[1].lower().split(".")[0],
    )

    # Load images in sorted order
    frames = []
    for _, filename in sorted_files:
        img_path = os.path.join(folder_path, filename)
        img = Image.open(img_path)

        # Create a white backdrop image
        backdrop = Image.new("RGBA", img.size, (255, 255, 255, 255))
        # Paste the original image onto the backdrop
        backdrop.paste(img, (0, 0), img)
        frames.append(backdrop.convert("P"))

    # Calculate duration if not provided
    if duration is None:
        # Adjust duration based on number of frames
        num_frames = len(frames)
        duration = min(50, 1000 // num_frames)  # Ensure minimum duration of 50ms

    # Save as GIF
    if len(frames) > 0:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
        logger.info(
            f"Successfully created GIF at {output_path} with {len(frames)} frames"
        )
        logger.info(f"Duration per frame: {duration}ms")
    else:
        logger.info("No valid PNG files found to create GIF")


def main(path, dups_sorted, tmp_file_names):
    path_main_dir = os.path.basename(os.path.normpath(path))
    line_raw_path = osp.join(path, "line_raw")
    line_file = os.path.join(path, "line_raw.gif")

    res_raw_path = osp.join(path, "final")
    res_file = os.path.join(path, f"{path_main_dir}.gif")

    create_pngs(path, dups_sorted, tmp_file_names, "final")

    create_gif(folder_path=line_raw_path, output_path=line_file, duration=200)

    create_gif(folder_path=res_raw_path, output_path=res_file, duration=200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create GIF from PNG sequence")
    parser.add_argument("folder_path", help="Path to folder containing PNG files")
    parser.add_argument("--output", default="output.gif", help="Output GIF filename")
    parser.add_argument(
        "--duration",
        type=int,
        default=None,
        help="Frame duration in milliseconds (optional, auto-calculated if not provided)",
    )

    args = parser.parse_args()

    create_gif(args.folder_path, output_path=args.output, duration=args.duration)
