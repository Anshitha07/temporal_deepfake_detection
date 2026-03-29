import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.trag_dataset import TRAGDataset
from src.models.trag_tcn import TRAG_TCN
from src.utils.trag_utils import compute_trag_input

# ================= CONFIG =================
TRAG_ROOT   = "data/processed_sdfvd/trag"
CKPT_PATH   = "checkpoints/trag_tcn_sdfvd_best.pth"
OUTPUT_ROOT = "data/processed_sdfvd/trag_features"

BATCH_SIZE  = 1
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
# ==========================================


def main():
    print("[INFO] Using device:", DEVICE)

    os.makedirs(os.path.join(OUTPUT_ROOT, "real"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "fake"), exist_ok=True)

    dataset = TRAGDataset(TRAG_ROOT)
    loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TRAG_TCN().to(DEVICE)
    checkpoint = torch.load(CKPT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    model.classifier = torch.nn.Identity()
    model.eval()

    print("[INFO] Extracting TRAG features for SDFVD...")

    with torch.no_grad():
        for idx, (trag, label) in enumerate(tqdm(loader)):
            trag = trag.to(DEVICE)
            trag = compute_trag_input(trag)
            feature = model(trag)
            feature = feature.squeeze(0).cpu()

            trag_path  = dataset.samples[idx][0]
            video_name = os.path.basename(trag_path).replace(".npy", "")
            class_name = "real" if label.item() == 0 else "fake"

            save_path = os.path.join(OUTPUT_ROOT, class_name, f"{video_name}.pt")
            torch.save(feature, save_path)

    print("[DONE] SDFVD TRAG feature extraction complete")
    print("[INFO] Features saved to:", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
