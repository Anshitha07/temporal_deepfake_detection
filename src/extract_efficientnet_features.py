import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.clip_dataset import CLIPFrameDataset
from src.models.efficientnet_encoder import EfficientNetEncoder

# ================= CONFIG =================
FRAMES_ROOT = "data/processed/frames"
OUTPUT_ROOT = "data/processed/efficientnet_features"
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================


def main():
    os.makedirs(os.path.join(OUTPUT_ROOT, "real"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "fake"), exist_ok=True)

    dataset = CLIPFrameDataset(FRAMES_ROOT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = EfficientNetEncoder(device=DEVICE)

    with torch.no_grad():
        for idx, (frames, label) in enumerate(tqdm(loader, desc="Extracting EfficientNet features")):
            frames = frames.to(DEVICE)

            feat = model(frames).squeeze(0).cpu()  # (1280,)

            video_dir = dataset.samples[idx][0]
            video_id = os.path.basename(video_dir)
            cls = "real" if label == 0 else "fake"

            torch.save(
                feat,
                os.path.join(OUTPUT_ROOT, cls, f"{video_id}.pt")
            )

    print("[DONE] EfficientNet feature extraction complete")


if __name__ == "__main__":
    main()
