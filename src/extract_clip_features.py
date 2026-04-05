import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.clip_dataset import CLIPFrameDataset
from src.models.clip_encoder import CLIPVisualEncoder


# ================= CONFIG =================
FRAMES_ROOT = "data/processed_ffpp/faces"
OUTPUT_ROOT = "data/processed_ffpp/clip_features2"

BATCH_SIZE = 1
MAX_FRAMES = 32   # 🔥 increased for better temporal info
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

    print(f"[INFO] Total videos: {len(dataset)}")

    with torch.no_grad():

        for idx, (frames, label) in enumerate(
            tqdm(loader, desc="Extracting CLIP features")
        ):

            frames = frames.to(DEVICE)  # (1, T, 3, 224, 224)

            # ===============================
            # Step 1: Get per-frame features
            # ===============================
            features = model(frames)        # (1, T, 512)
            features = features.squeeze(0) # (T, 512)

            # ===============================
            # Step 2: Temporal aggregation
            # ===============================
            mean_feat = features.mean(dim=0)
            max_feat, _ = features.max(dim=0)

            # 🔥 FINAL FEATURE (1024 dim)
            final_feat = torch.cat([mean_feat, max_feat], dim=0)

            # ===============================
            # Save feature
            # ===============================
            video_dir, lbl = dataset.samples[idx]
            video_name = os.path.basename(video_dir)

            class_name = "real" if lbl == 0 else "fake"

            save_path = os.path.join(
                OUTPUT_ROOT,
                class_name,
                f"{video_name}.pt"
            )

            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            torch.save(final_feat.cpu(), save_path)

    print("[DONE] Feature extraction complete")


def main():
    print("[INFO] Using device:", DEVICE)
    process_split()
    print(f"[INFO] Features saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()