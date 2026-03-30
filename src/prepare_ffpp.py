import os
import cv2
import glob
import random
from pathlib import Path


def extract_frames_from_video(video_path, out_dir, fps_skip=1):
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video {video_path}")

    frame_id = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_id % fps_skip == 0:
            out_path = os.path.join(out_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1

        frame_id += 1

    cap.release()
    return saved


def prepare_frames(root_videos, output_root, fps_skip=1):
    """
    FF++ structure:
        original → REAL
        Deepfakes, FaceSwap, FaceShifter → FAKE
    """

    if not os.path.exists(root_videos):
        raise FileNotFoundError(f"Dataset path not found: {root_videos}")

    print(f"[INFO] Using FF++ dataset: {root_videos}")

    # 🔥 mapping
    folders = [
        ("original", "real"),
        ("Deepfakes", "fake"),
        ("FaceSwap", "fake"),
        ("FaceShifter", "fake"),
    ]

    for folder_name, label in folders:

        folder_path = os.path.join(root_videos, folder_name)

        if not os.path.exists(folder_path):
            print(f"[SKIP] {folder_name} not found")
            continue

        videos = sorted(glob.glob(os.path.join(folder_path, "*.mp4")))

        print(f"[INFO] Found {len(videos)} videos in {folder_name}")

        for video in videos:
            video_id = Path(video).stem

            out_video_dir = os.path.join(output_root, label, video_id)
            os.makedirs(out_video_dir, exist_ok=True)

            # resume support
            if len(os.listdir(out_video_dir)) > 0:
                print(f"[SKIP] {video_id}")
                continue

            print(f"[PROCESS] {folder_name}/{video_id}")

            n = extract_frames_from_video(video, out_video_dir, fps_skip=fps_skip)

            if n == 0:
                print(f"[WARNING] No frames extracted for {video}")


def create_split_lists(output_root, train_ratio=0.8, seed=42):

    random.seed(seed)
    all_pairs = []

    for label in ["real", "fake"]:
        label_folder = os.path.join(output_root, label)

        if not os.path.exists(label_folder):
            continue

        for video_dir in os.listdir(label_folder):
            path = os.path.join(label_folder, video_dir)
            if os.path.isdir(path):
                all_pairs.append((label, video_dir))

    if len(all_pairs) == 0:
        raise RuntimeError("No videos found. Run frame extraction first.")

    random.shuffle(all_pairs)

    n_train = int(train_ratio * len(all_pairs))

    train_pairs = all_pairs[:n_train]
    test_pairs = all_pairs[n_train:]

    with open(os.path.join(output_root, "split_train.txt"), "w") as f:
        for label, vid in train_pairs:
            f.write(f"{label}/{vid}\n")

    with open(os.path.join(output_root, "split_test.txt"), "w") as f:
        for label, vid in test_pairs:
            f.write(f"{label}/{vid}\n")

    print(f"[INFO] Train: {len(train_pairs)} | Test: {len(test_pairs)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare FF++ dataset")

    parser.add_argument(
        "--video_root",
        default="data/ffpp_raw",
        help="FF++ dataset root (original, Deepfakes, etc.)"
    )

    parser.add_argument(
        "--out_root",
        default="data/processed_ffpp/frames",
        help="output frames directory"
    )

    parser.add_argument("--fps_skip", type=int, default=5)
    parser.add_argument("--split_train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    prepare_frames(args.video_root, args.out_root, fps_skip=args.fps_skip)
    create_split_lists(args.out_root, args.split_train_ratio, args.seed)

    print("\n[DONE] FF++ frames prepared")