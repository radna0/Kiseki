import cv2
import numpy as np


def main(video_path: str):
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Couldn't open video {video_path}")
        return

    # Read first frame and initialize previous gray frame
    ret, frame1 = cap.read()
    if not ret:
        print("Error: Couldn't read first frame.")
        return
    prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Prepare HSV image for visualization
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255  # set saturation to maximum

    while True:
        ret, frame2 = cap.read()
        if not ret:
            break  # end of video

        next_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

        # Calculate dense optical flow using Farneback’s algorithm
        flow = cv2.calcOpticalFlowFarneback(
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
        )

        # Compute magnitude and angle of 2D vectors
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

        # Hue corresponds to flow direction (angle)
        hsv[..., 0] = ang * (180 / np.pi / 2)
        # Value corresponds to flow magnitude (normalized)
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

        # Convert HSV to BGR for display
        bgr_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        # Show the flow visualization
        cv2.imshow("Dense Optical Flow", bgr_flow)

        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC to exit
            break
        elif key == ord("s"):
            # Save current frame and flow visualization
            cv2.imwrite("optical_frame.png", frame2)
            cv2.imwrite("optical_flow.png", bgr_flow)
            print("Saved optical_frame.png and optical_flow.png")

        # Update previous frame
        prvs = next_gray

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import sys

    video_file = sys.argv[1] if len(sys.argv) > 1 else "vtest.avi"
    main(video_file)
