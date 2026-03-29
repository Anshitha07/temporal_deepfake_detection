import sys
import os
import cv2
from pathlib import Path

# Add project root to sys.path so package imports from src work when running as script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.face_preprocessing import detect_face


def extract_faces(frame_dir, output_dir, img_size=224):
    os.makedirs(output_dir, exist_ok=True)

    frames = sorted(
        f for f in os.listdir(frame_dir)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    )

    if len(frames) == 0:
        print(f"[WARN] no frames in {frame_dir}")
        return 0

    saved = 0

    for frame_name in frames:
        frame_path = os.path.join(frame_dir, frame_name)
        img = cv2.imread(frame_path)

        if img is None:
            continue

        face = detect_face(img)
        if face is None:
            continue

        face = cv2.resize(face, (img_size, img_size))
        out_file = os.path.join(output_dir, f"{saved:03d}.jpg")
        cv2.imwrite(out_file, face)
        saved += 1

    return saved


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="Extract faces from CelebDF frame folders")
    parser.add_argument("--input_dir", default="data/processed_celebdf/frames", help="source frames root (real/fake/video_id)")
    parser.add_argument("--output_dir", default="data/processed_celebdf/faces", help="face crops output root")
    parser.add_argument("--img_size", type=int, default=224, help="face crop size (square)")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dir not found: {input_dir}")

    for label in ["real", "fake"]:
        label_dir = input_dir / label
        if not label_dir.exists():
            print(f"[WARN] missing folder: {label_dir}")
            continue

        for video_dir in sorted(label_dir.iterdir()):
            if not video_dir.is_dir():
                continue

            vid_name = video_dir.name
            save_dir = os.path.join(args.output_dir, label, vid_name)

            # skip if already extracted
            if os.path.exists(save_dir) and any(f.lower().endswith(('.jpg', '.png', '.jpeg')) for f in os.listdir(save_dir)):
                print(f"[{label.upper()}] {vid_name} (skip; already processed)")
                continue

            print(f"[{label.upper()}] {vid_name}")
            saved = extract_faces(str(video_dir), save_dir, img_size=args.img_size)
            print(f"  saved {saved} face frames")
