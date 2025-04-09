import argparse
import time
import utils.deduplicate as dedup
from utils.logging import Profiler
import utils.segmentation as seg
import utils.coloring as col
import utils.sequence as seq


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="dataset/test/laughing_girl",
        help="path to your anime clip folder or folder containing multiple clips.",
    )
    parser.add_argument(
        "--mode",
        choices=["nearest", "reference"],
        default="nearest",
        help="",
    )
    parser.add_argument(
        "--skip_seg", action="store_true", help="used when `seg` already exists."
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=4,
        help="used together with `--seg_type trappedball`. Increase the value if unclosed pixels' high.",
    )
    parser.add_argument(
        "--multi_clip",
        action="store_true",
        help="used for multi-clip inference. Set `path` to a folder where each sub-folder is a single clip.",
    )
    parser.add_argument(
        "--keep_line",
        action="store_true",
        help="used for keeping the original line in the final output.",
    )
    parser.add_argument(
        "--raft_res",
        type=int,
        default=1024,
        help="change the resolution for the optical flow estimation. If the performance is bad on your case, you can change this to 1024 to have a try.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Deduplicate frames
    with Profiler("Dedup Time", limit=5):
        dups_sorted, tmp_file_names = dedup.main(args.path)

    # Segmentation
    with Profiler("Segment Time", limit=5):
        if not args.skip_seg:
            seg.main(args)

    # raise NotImplementedError

    # Coloring
    with Profiler("Coloring Time", limit=5):
        col.main(args)

    # Sequence
    with Profiler("Sequence Time", limit=5):
        seq.main(args.path, dups_sorted, tmp_file_names)


if __name__ == "__main__":
    with Profiler("Main Time", limit=5):
        main()
