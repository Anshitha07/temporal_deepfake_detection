import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import csv
import numpy as np

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from src.models.fusion_model import FusionModel


# ================= CONFIG =================
TRAG_FEAT_ROOT = "data/processed_celebdf/trag_features"
CLIP_FEAT_ROOT = "data/processed_celebdf/clip_features"

CHECKPOINT = "checkpoints/fusion_best_celebdf.pth"

BATCH_SIZE = 8
DEVICE = "cpu"

CROSS_DATASET = "CelebDF intra"
# ==========================================


class CrossFusionDataset(Dataset):

    def __init__(self, trag_root, clip_root):

        self.samples = []

        splits = ["train", "val", "test"]  # 🔥 include ALL

        for split in splits:
            for label_name, label in [("real", 0), ("fake", 1)]:

                trag_dir = os.path.join(trag_root, split, label_name)
                clip_dir = os.path.join(clip_root, split, label_name)

                if not os.path.exists(trag_dir) or not os.path.exists(clip_dir):
                    continue

                trag_files = [f for f in os.listdir(trag_dir) if f.endswith(".pt")]
                clip_files = set(os.listdir(clip_dir))

                for f in trag_files:
                    if f in clip_files:
                        self.samples.append((
                            os.path.join(trag_dir, f),
                            os.path.join(clip_dir, f),
                            label
                        ))

        print("[INFO] Total samples:", len(self.samples))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        trag_path, clip_path, label = self.samples[idx]

        trag_feat = torch.load(trag_path).float()
        clip_feat = torch.load(clip_path).float()

        return trag_feat, clip_feat, torch.tensor(label), trag_path


# ================= SAFE LOAD =================
def load_checkpoint_safely(model, checkpoint_path, device):
    print("\n[INFO] Loading checkpoint safely...")

    ckpt = torch.load(checkpoint_path, map_location=device)

    if "fusion_model" in ckpt:
        ckpt = ckpt["fusion_model"]
    elif "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    elif "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            k = k[7:]
        new_ckpt[k] = v

    model_dict = model.state_dict()
    filtered_ckpt = {}

    for k, v in new_ckpt.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_ckpt[k] = v
        else:
            print(f"[SKIPPED] {k} -> {v.shape}")

    model_dict.update(filtered_ckpt)
    model.load_state_dict(model_dict)

    print(f"[INFO] Loaded {len(filtered_ckpt)} layers\n")


# ================= MAIN =================
def main():

    print("[INFO] Using device:", DEVICE)

    dataset = CrossFusionDataset(TRAG_FEAT_ROOT, CLIP_FEAT_ROOT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = FusionModel().to(DEVICE)

    load_checkpoint_safely(model, CHECKPOINT, DEVICE)

    model.eval()

    correct = 0
    total = 0

    results = []
    all_labels = []
    all_probs = []

    with torch.no_grad():

        for i, (trag_feat, clip_feat, labels, paths) in enumerate(tqdm(loader)):

            trag_feat = trag_feat.to(DEVICE)
            clip_feat = clip_feat.to(DEVICE)
            labels = labels.to(DEVICE)

            if i == 0:
                print("[DEBUG] TRAG:", trag_feat.shape)
                print("[DEBUG] CLIP:", clip_feat.shape)

            logits, gates = model(trag_feat, clip_feat, return_gate=True)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            for j in range(len(paths)):
                results.append([
                    os.path.basename(paths[j]),
                    labels[j].item(),
                    preds[j].item(),
                    probs[j].item(),
                    gates[j].item()
                ])

    # ================= METRICS =================
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = (all_probs >= 0.5).astype(int)

    acc = correct / total if total > 0 else 0
    roc_auc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    # ================= SAVE =================
    os.makedirs("results", exist_ok=True)

    output_file = "results/celebdf_fusion_predictions.csv"

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "ground_truth", "prediction", "confidence", "gate"])

        for r in results:
            writer.writerow(r)

    print("\n========== CROSS DATASET RESULT ==========")
    print("Dataset:", CROSS_DATASET)

    print(f"\nAccuracy  : {acc:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nConfusion Matrix:\n", cm)

    print("\n[INFO] Predictions saved to:", output_file)
    print("[DONE]")


if __name__ == "__main__":
    main()