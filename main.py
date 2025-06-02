import argparse
import time
import kiseki.arcs.deduplicate as dedup
import kiseki.arcs.segmentation as seg
import kiseki.arcs.coloring as col
import kiseki.arcs.sequence as seq

# import kiseki.arcs.lineart as lineart
from kiseki.logging import Profiler


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
        default="reference",
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
        default=640,
        help="change the resolution for the optical flow estimation. If the performance is bad on your case, you can change this to 640 to have a try.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Step 1.A: Deduplicate frames
    with Profiler("Dedup Time", limit=0):
        raw_ref_map, dups_sorted, tmp_file_names, raw_line_map = dedup.main(args.path)

    """ # Step 1.B: Process images for transparency if needed
    with Profiler("Transparency Processing Time", limit=5):
        lineart.main(args) """

    """ # Step 1.C: Cleanup Linearts
    with Profiler("Cleanup Lineart Time", limit=0):
        cleanup.main(args) """

    # Step 2: Segmentation
    with Profiler("Segment Time", limit=0):
        if not args.skip_seg:
            seg.main(args)

    # raise NotImplementedError

    # Step 3: Coloring
    with Profiler("Coloring Time", limit=5):
        col.main(args)

    # Step 4: Sequence
    with Profiler("Sequence Time", limit=0):
        seq.main(args.path, dups_sorted, tmp_file_names)


if __name__ == "__main__":
    with Profiler("Main Time", limit=0):
        main()
