import argparse
import os
import time
import kiseki.arcs.deduplicate as dedup
import kiseki.arcs.segmentation as seg
import kiseki.arcs.coloring as col
import kiseki.arcs.sequence as seq

# import kiseki.arcs.lineart as lineart
from kiseki.logging import Profiler, setup_config, logger
import requests

from flask import Flask, request, jsonify
from werkzeug.serving import make_server
import sys
import threading

sys.stdout.reconfigure(encoding="utf-8")

app = Flask(__name__)
MODEL_INFERENCE = None


# This Event will let the main thread know that Flask is now serving.
server_ready = threading.Event()


class ServerThread(threading.Thread):
    def __init__(self, args):
        super().__init__(daemon=True)
        self.args = args
        self.server = None

    def run(self):
        global MODEL_INFERENCE
        setup_config()
        MODEL_INFERENCE = col.init(self.args)

        # Create a Werkzeug “server” object, tell it to listen on 127.0.0.1:8000.
        self.server = make_server("127.0.0.1", 8000, app, processes=os.cpu_count())
        # Once we’ve bound the socket, set the event so main() can proceed.
        server_ready.set()

        # Now run the server forever (or until killed).
        self.server.serve_forever()

    def shutdown(self):
        if self.server:
            self.server.shutdown()


@app.route("/inference", methods=["GET"])
def inference():
    global MODEL_INFERENCE
    path = request.args["path"]
    MODEL_INFERENCE.inference_multi_gt_sequential(path)
    return jsonify({"message": "Done!"}), 200


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
    return parser.parse_args()


def main(args):
    global MODEL_INFERENCE
    # Step 1.A: Deduplicate frames
    with Profiler("Dedup Time", limit=0):
        _, dups_sorted, tmp_file_names, __ = dedup.main(args)

    print(args)

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
    with Profiler("Coloring Time", limit=20):
        while True:
            if not server_ready.is_set():
                continue
            logger.info(f"Flask server started!")
            res = requests.get(f"http://127.0.0.1:8000/inference?path={args.path}")
            logger.info(f"res: {res}")
            break

    # Step 4: Sequence
    with Profiler("Sequence Time", limit=0):
        seq.main(args.path, dups_sorted, tmp_file_names)


if __name__ == "__main__":
    with Profiler("Main Time", limit=0):
        args = parse_args()
        flask_thread = ServerThread(args)
        flask_thread.start()
        main(args)