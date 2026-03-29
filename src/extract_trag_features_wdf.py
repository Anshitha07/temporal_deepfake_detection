import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.trag_dataset_wdf import TRAGDataset
from src.models.trag_tcn import TRAG_TCN


# ================= CONFIG =================
TRAG_ROOT = "data/processed_uadfv/trag"

CHECKPOINT = "checkpoints/trag_tcn_best.pth"

OUTPUT_ROOT = "data/processed_uadfv/trag_features"

BATCH_SIZE = 1
DEVICE = "cpu"   # run on CPU
# ==========================================


def main():

    print("[INFO] Using device:", DEVICE)

    os.makedirs(os.path.join(OUTPUT_ROOT, "real"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_ROOT, "fake"), exist_ok=True)

    dataset = TRAGDataset(TRAG_ROOT)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    model = TRAG_TCN().to(DEVICE)

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    model.classifier = torch.nn.Identity()
    model.eval()

    print("[INFO] Extracting TRAG features for UADFV...")

    with torch.no_grad():

        for idx,(trag,label) in enumerate(tqdm(loader)):

            trag = trag.to(DEVICE)

            B,T,C,H,W = trag.shape

            trag = trag.mean(dim=(3,4))

            zero = torch.zeros(B,T,1,device=trag.device)

            trag = torch.cat((trag,zero),dim=2)

            feature = model(trag)

            feature = feature.squeeze(0).cpu()

            trag_path = dataset.samples[idx][0]

            video_name = os.path.basename(trag_path).replace(".npy","")

            class_name = "real" if label.item()==0 else "fake"

            save_path = os.path.join(
                OUTPUT_ROOT,
                class_name,
                f"{video_name}.pt"
            )

            torch.save(feature,save_path)

    print("[DONE] UADFV TRAG feature extraction complete")


if __name__ == "__main__":
    main()