import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.trag_dataset import TRAGDataset
from src.models.trag_tcn import TRAG_TCN

# ================= CONFIG =================
TRAG_ROOT   = "data/processed_ffpp/trag"
CKPT_PATH   = "checkpoints/trag_tcn_best.pth"
OUTPUT_ROOT = "data/processed_ffpp/trag_features"
BATCH_SIZE  = 1
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================


def main():
    print("[INFO] Using device:", DEVICE)

    os.makedirs(os.path.join(OUTPUT_ROOT, "real"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "fake"), exist_ok=True)

    dataset = TRAGDataset(TRAG_ROOT)
    print(f"[INFO] Loaded TRAG dataset ({len(dataset)} samples)")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0)

    model = TRAG_TCN().to(DEVICE)
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    # Remove classifier — keep 128-dim features
    model.classifier = torch.nn.Identity()
    model.eval()

    print("[INFO] Extracting TRAG features...")
    real_count, fake_count = 0, 0

    with torch.no_grad():
        progress_bar = tqdm(loader, desc="Extracting Features", unit="video")

        for idx, (trag, label) in enumerate(progress_bar):
            trag = trag.to(DEVICE)

            B, T, C, H, W = trag.shape
            trag = trag.mean(dim=(3, 4))
            zero = torch.zeros(B, T, 1, device=trag.device)
            trag = torch.cat((trag, zero), dim=2)

            feature = model(trag)
            feature = feature.squeeze(0).cpu()

            trag_path  = dataset.samples[idx][0]
            video_name = os.path.basename(trag_path).replace(".npy", "")
            class_name = "real" if label.item() == 0 else "fake"

            if class_name == "real":
                real_count += 1
            else:
                fake_count += 1

            save_path = os.path.join(OUTPUT_ROOT, class_name, f"{video_name}.pt")

            if os.path.exists(save_path):
                continue

            torch.save(feature, save_path)
            progress_bar.set_postfix({"Real": real_count, "Fake": fake_count})

    print("\n[DONE] TRAG feature extraction complete")
    print(f"[INFO] Features saved to: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()