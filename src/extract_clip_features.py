import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.clip_dataset import CLIPFrameDataset
from src.models.clip_encoder import CLIPVisualEncoder


# ================= CONFIG =================
# For CelebDF-small
FRAMES_ROOT = "data/processed_ffpp/frames"
OUTPUT_ROOT = "data/processed_ffpp/clip_features"

BATCH_SIZE = 1
MAX_FRAMES = 16
DEVICE = "cpu"  # force CPU for local machines; change to cuda if available
# ==========================================


def process_split():

    dataset = CLIPFrameDataset(
        frames_root=FRAMES_ROOT,
        max_frames=MAX_FRAMES
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    model = CLIPVisualEncoder(device=DEVICE).to(DEVICE)
    model.eval()

    with torch.no_grad():

        for idx, (frames, label) in enumerate(
            tqdm(loader, desc="Extracting CLIP features)")
        ):

            frames = frames.to(DEVICE)

            features = model(frames)
            features = features.squeeze(0).cpu()

            video_dir, lbl = dataset.samples[idx]
            video_name = os.path.basename(video_dir)

            class_name = "real" if lbl == 0 else "fake"

            save_path = os.path.join(
                OUTPUT_ROOT,
                class_name,
                f"{video_name}.pt"
            )

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.save(features, save_path)


def main():

    print("[INFO] Using device:", DEVICE)

    process_split()

    print("[DONE] CLIP feature extraction complete")
    print(f"[INFO] Features saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()