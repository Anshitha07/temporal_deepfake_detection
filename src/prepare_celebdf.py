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
    """Extract frames for all real/fake videos in root_videos into output_root."""
    # if user passes a location that doesn't exist, try src/data fallback
    if not os.path.exists(root_videos):
        fallback = os.path.join("src", "data", "celeb_df_small")
        if os.path.exists(fallback):
            print(f"[INFO] via --video_root not found, using fallback {fallback}")
            root_videos = fallback
        else:
            raise FileNotFoundError(
                f"Video root not found: {root_videos}. "
                f"Fallback also not found: {fallback}."
            )

    for label in ["real", "fake"]:
        videos = sorted(glob.glob(os.path.join(root_videos, label, "*.mp4")))
        print(f"Found {len(videos)} {label} videos")

        for video in videos:
            video_id = Path(video).stem
            out_video_dir = os.path.join(output_root, label, video_id)
            os.makedirs(out_video_dir, exist_ok=True)

            if os.path.exists(out_video_dir) and len(os.listdir(out_video_dir)) > 0:
                print(f"Skipped {video_id} (already extracted)")
                continue

            print(f"Extracting {label}/{video_id}")
            n = extract_frames_from_video(video, out_video_dir, fps_skip=fps_skip)
            if n == 0:
                print(f"WARNING: no frames extracted for {video}")


def create_split_lists(output_root, train_ratio=0.8, seed=42):
    """Split the extracted video folders into train/test text lists"""
    os.makedirs(output_root, exist_ok=True)

    random.seed(seed)
    all_pairs = []  # (label, video_id)
    for label in ["real", "fake"]:
        label_folder = os.path.join(output_root, label)
        if not os.path.exists(label_folder):
            continue

        for video_dir in sorted(glob.glob(os.path.join(label_folder, "*"))):
            if os.path.isdir(video_dir):
                video_id = os.path.basename(video_dir)
                all_pairs.append((label, video_id))

    random.shuffle(all_pairs)

    if len(all_pairs) == 0:
        raise RuntimeError(
            f"No extracted video directories found under {output_root}. "
            "Please run prepare_frames first and verify videos exist."
        )

    n_train = int(train_ratio * len(all_pairs))
    train_pairs = all_pairs[:n_train]
    test_pairs = all_pairs[n_train:]

    with open(os.path.join(output_root, "split_train.txt"), "w") as f:
        for label, vid in train_pairs:
            f.write(f"{label}/{vid}\n")

    with open(os.path.join(output_root, "split_test.txt"), "w") as f:
        for label, vid in test_pairs:
            f.write(f"{label}/{vid}\n")

    print(f"Saved split_train.txt ({len(train_pairs)}) and split_test.txt ({len(test_pairs)})")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare CelebDF-small frames for pipeline")
    parser.add_argument("--video_root", default="data/celeb_df_small", help="raw mp4 root (real/fake)")
    parser.add_argument("--out_root", default="data/processed_celebdf/frames", help="frames output root")
    parser.add_argument("--fps_skip", type=int, default=1, help="keep every Nth frame")
    parser.add_argument("--split_train_ratio", type=float, default=0.8, help="train ratio")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()

    prepare_frames(args.video_root, args.out_root, fps_skip=args.fps_skip)
    create_split_lists(args.out_root, train_ratio=args.split_train_ratio, seed=args.seed)

    print("Done preparing CelebDF frames.")
