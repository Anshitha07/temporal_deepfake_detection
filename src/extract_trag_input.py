import os
import numpy as np
import cv2


def frames_to_trag_npy(input_root, output_root, max_frames=None, image_size=224):
    os.makedirs(output_root, exist_ok=True)

    for label in ["real", "fake"]:
        in_label_dir = os.path.join(input_root, label)
        out_label_dir = os.path.join(output_root, label)

        if not os.path.isdir(in_label_dir):
            print(f"[WARNING] Missing folder: {in_label_dir}")
            continue

        os.makedirs(out_label_dir, exist_ok=True)

        print(f"\n[INFO] Processing {label} videos...")

        for video_id in sorted(os.listdir(in_label_dir)):
            video_dir = os.path.join(in_label_dir, video_id)

            if not os.path.isdir(video_dir):
                continue

            out_path = os.path.join(out_label_dir, f"{video_id}.npy")

            # ✅ Resume support
            if os.path.exists(out_path):
                continue

            frame_files = sorted(
                f for f in os.listdir(video_dir)
                if f.lower().endswith(('.jpg', '.png', '.jpeg'))
            )

            if len(frame_files) == 0:
                continue

            if max_frames is not None:
                frame_files = frame_files[:max_frames]

            print(f"Processing: {label}/{video_id} ({len(frame_files)} frames)")

            frames = []

            for frame_name in frame_files:
                p = os.path.join(video_dir, frame_name)

                img = cv2.imread(p)
                if img is None:
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (image_size, image_size))

                frames.append(img)

            if len(frames) == 0:
                continue

            arr = np.stack(frames, axis=0)
            arr = arr.transpose(0, 3, 1, 2)

            np.save(out_path, arr)

    print(f"\n[DONE] Generated TRAG inputs at {output_root}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract TRAG input arrays from frames")

    # 🔥 IMPORTANT CHANGE HERE
    parser.add_argument(
        "--input_root",
        default="data/processed_celebdf/frames",  # ✅ using frames now
        help="frame root with real/fake dirs"
    )

    parser.add_argument(
        "--output_root",
        default="data/processed_celebdf/trag",
        help="output npy root"
    )

    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=224)

    args = parser.parse_args()

    frames_to_trag_npy(
        args.input_root,
        args.output_root,
        args.max_frames,
        args.image_size
    )