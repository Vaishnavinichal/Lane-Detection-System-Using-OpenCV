# run.py
import argparse
import cv2
import os
from lane_detector import LaneDetector

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', default='test2.mp4', help='input video file or camera index (default test2.mp4)')
    p.add_argument('--output', '-o', default='lane_out.avi', help='output video filename (optional)')
    p.add_argument('--display', '-d', action='store_true', help='show windows')
    p.add_argument('--debug', action='store_true', help='save sample debug images')
    return p.parse_args()

def main():
    args = parse_args()
    inp = args.input
    out = args.output if args.output else None

    # choose capture
    is_camera = False
    if inp.isdigit():
        cap = cv2.VideoCapture(int(inp), cv2.CAP_DSHOW)
        is_camera = True
    else:
        cap = cv2.VideoCapture(inp)

    if not cap.isOpened():
        print("ERROR: cannot open input:", inp)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input: {inp} | size: {w}x{h} | fps: {fps}")

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    writer = None
    if out:
        writer = cv2.VideoWriter(out, fourcc, fps, (w, h))
        print("Writing output to:", out)

    detector = LaneDetector(debug=args.debug)
    params = {
        'blur_ksize': 5,
        'canny_low': 50,
        'canny_high': 150,
        # optional: compute ROI per frame using default helper
    }

    frame_idx = 0
    saved_samples = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of stream or cannot fetch frame.")
                break
            frame_idx += 1

            overlay, debug_info = detector.process(frame, params)

            if writer:
                writer.write(overlay)

            if args.display:
                cv2.imshow('Original', frame)
                cv2.imshow('Overlay', overlay)
                cv2.imshow('Edges', debug_info['edges'])
                cv2.imshow('Masked', debug_info['masked'])

            # save sample images once
            if args.debug and not saved_samples:
                cv2.imwrite('dbg_edges.jpg', debug_info['edges'])
                cv2.imwrite('dbg_masked.jpg', debug_info['masked'])
                cv2.imwrite('dbg_overlay.jpg', overlay)
                saved_samples = True
                print("Saved dbg_* images")

            # pause on first frame for inspection (helpful)
            if frame_idx == 1 and args.display:
                print("Paused on first frame. Press any key in a window to continue...")
                cv2.waitKey(0)

            if args.display:
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Exit requested by user.")
                    break

    finally:
        cap.release()
        if writer:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()
        print("Finished. Processed frames:", frame_idx)

if __name__ == '__main__':
    main()
