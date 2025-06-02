import os
import sys
import logging
import cv2
import numpy as np
from pathlib import Path


# Configure logging
def setup_logging(log_file="flow_processing.log"):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )


# Sparse feature initialization
def init_sparse_lk(frame_gray):
    return cv2.goodFeaturesToTrack(
        frame_gray, maxCorners=200, qualityLevel=0.01, minDistance=7
    )


# Draw sparse flow tracks
def draw_sparse_flow(p0, p1, mask, frame, colors):
    for i, (new, old) in enumerate(zip(p1, p0)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), colors[i].tolist(), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, colors[i].tolist(), -1)
    return mask, frame


# Convert flow to BGR color
def _flow_to_color(flow, shape):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*shape, 3), np.uint8)
    hsv[..., 1] = 255
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


# Dense flow methods


def dense_flow_farneback(prvs, next_gray):
    return _flow_to_color(
        cv2.calcOpticalFlowFarneback(
            prvs,
            next_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        ),
        prvs.shape,
    )


def dense_flow_dis(prvs, next_gray, dis):
    return _flow_to_color(dis.calc(prvs, next_gray, None), prvs.shape)


def dense_flow_tvl1(prvs, next_gray, tvl1):
    return _flow_to_color(tvl1.calc(prvs, next_gray, None), prvs.shape)


# Main processing
def main(video_path="vtest.avi", output="flow_comparison", codec="X264"):
    setup_logging()
    logging.info(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open {video_path}")
        return

    # Prepare flow algorithms
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    tvl1 = cv2.optflow.DualTVL1OpticalFlow_create()
    rng = np.random.default_rng(12345)
    colors = rng.integers(0, 255, (200, 3), dtype=np.uint8)

    # Read first frame
    ret, frame1 = cap.read()
    if not ret:
        logging.error("Failed to read first frame.")
        return

    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    p0 = init_sparse_lk(prvs)
    mask = np.zeros_like(frame1)

    h, w = frame1.shape[:2]
    out_w, out_h = w * 2, h * 2
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Decide whether to use video writer or image sequence
    use_video = True
    fourcc = cv2.VideoWriter_fourcc(*codec) if codec else 0
    try:
        out = cv2.VideoWriter(f"{output}.mp4", fourcc, fps, (out_w, out_h))
        if not out.isOpened():
            raise ValueError("VideoWriter failed to open.")
        logging.info(
            f"Writing output video: {output}.mp4, size: {(out_w, out_h)}, fps: {fps}"
        )
    except Exception as e:
        logging.warning(f"VideoWriter unavailable for resolution {out_w}x{out_h}: {e}")
        use_video = False
        # Create output folder for frames
        frames_dir = Path(f"{output}_frames")
        frames_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Falling back to image sequence in {frames_dir}")

    frame_idx = 0
    while True:
        ret, frame2 = cap.read()
        if not ret:
            logging.info("End of video reached, restarting.")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame2 = cap.read()
            if not ret:
                logging.info("No frames after restart. Exiting.")
                break
            prvs = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            p0 = init_sparse_lk(prvs)
            mask = np.zeros_like(frame2)

        next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        logging.debug(f"Processing frame {frame_idx}")

        # 1) Sparse Lucas-Kanade
        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prvs,
            next_gray,
            p0,
            None,
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        good_old = p0[st == 1]
        good_new = p1[st == 1]
        lk_mask = mask.copy()
        lk_frame = frame2.copy()
        if len(good_new) > 0:
            lk_mask, lk_frame = draw_sparse_flow(
                good_old, good_new, lk_mask, lk_frame, colors
            )

        # 2) Dense Farneback
        fb = dense_flow_farneback(prvs, next_gray)

        # 3) Dense DIS
        dis_img = dense_flow_dis(prvs, next_gray, dis)

        # 4) Dense Dual-TV L1
        tvl1_img = dense_flow_tvl1(prvs, next_gray, tvl1)

        # Stack & label
        top = np.hstack([lk_frame, fb])
        bottom = np.hstack([dis_img, tvl1_img])
        combined = np.vstack([top, bottom])
        cv2.putText(
            combined,
            "Sparse LK",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            combined,
            "Farneback",
            (w + 10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            combined,
            "DIS",
            (10, h + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            combined,
            "Dual-TV L1",
            (w + 10, h + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
        )

        # Output
        if use_video:
            out.write(combined)
            logging.info(f"Wrote frame {frame_idx} to video.")
        else:
            frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
            cv2.imwrite(str(frame_path), combined)
            logging.info(f"Saved frame {frame_idx} to {frame_path}")

        prvs = next_gray
        p0 = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else init_sparse_lk(prvs)
        mask = lk_mask
        frame_idx += 1

    # Cleanup
    cap.release()
    if use_video:
        out.release()
    logging.info("Processing completed and resources released.")


if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "vtest.avi"
    out_name = sys.argv[2] if len(sys.argv) > 2 else "flow_comparison"
    codec_arg = sys.argv[3] if len(sys.argv) > 3 else "X264"
    main(video, out_name, codec_arg)
