import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

from src.models.fusion_model import FusionModel


# ================= CONFIG =================
TRAG_FEAT_ROOT = "data/processed_celebdf/trag_features"
CLIP_FEAT_ROOT = "data/processed_celebdf/clip_features"

CHECKPOINT = "checkpoints/fusion_best_celebdf.pth"
VAL_INDICES_PATH = "checkpoints/fusion_val_indices_celebdf.json"

BATCH_SIZE = 32
DEVICE = "cpu"
SEED = 42
# =========================================


# ================= DATASET =================
class FusionDataset(Dataset):

    def __init__(self, trag_root, clip_root):

        self.samples = []

        for label_name, label in [("real", 0), ("fake", 1)]:

            trag_dir = os.path.join(trag_root, label_name)
            clip_dir = os.path.join(clip_root, label_name)

            if not os.path.exists(trag_dir) or not os.path.exists(clip_dir):
                print(f"[WARNING] Missing folder: {label_name}")
                continue

            trag_files = sorted([f for f in os.listdir(trag_dir) if f.endswith(".pt")])
            clip_files = sorted([f for f in os.listdir(clip_dir) if f.endswith(".pt")])

            common_files = sorted(list(set(trag_files) & set(clip_files)))

            for f in common_files:
                self.samples.append((
                    os.path.join(trag_dir, f),
                    os.path.join(clip_dir, f),
                    label
                ))

        print(f"[INFO] Total matched samples: {len(self.samples)}")

        if len(self.samples) == 0:
            raise RuntimeError("No matching TRAG + CLIP feature pairs found!")


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        trag_path, clip_path, label = self.samples[idx]

        trag_feat = torch.load(trag_path)
        clip_feat = torch.load(clip_path)

        return (
            trag_feat.float(),
            clip_feat.float(),
            torch.tensor(label, dtype=torch.long),
            trag_path
        )


# ================= SAFE LOAD =================
def load_checkpoint_safely(model, checkpoint_path, device):
    print("[INFO] Loading checkpoint...")

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

    # 🔥 SAFE FILTER (important)
    model_dict = model.state_dict()
    filtered_ckpt = {}

    for k, v in new_ckpt.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_ckpt[k] = v
        else:
            print(f"[SKIPPED] {k} -> {v.shape}")

    model_dict.update(filtered_ckpt)
    model.load_state_dict(model_dict)

    print(f"[INFO] Loaded {len(filtered_ckpt)} layers safely")


# ================= MAIN =================
def main():

    torch.manual_seed(SEED)

    print("[INFO] Using device:", DEVICE)

    # ===== LOAD FULL DATASET =====
    full_dataset = FusionDataset(TRAG_FEAT_ROOT, CLIP_FEAT_ROOT)

    # ===== 80/20 SPLIT (deterministic) =====
    if not os.path.exists(VAL_INDICES_PATH):
        raise FileNotFoundError(
            f"Val indices file not found: {VAL_INDICES_PATH}. "
            "Run train_fusion.py first."
        )

    import json
    with open(VAL_INDICES_PATH, "r") as f:
        val_indices = json.load(f)

    val_set = torch.utils.data.Subset(full_dataset, val_indices)

    print(f"[INFO] Validation samples: {len(val_set)}")

    loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

    # ===== MODEL =====
    model = FusionModel().to(DEVICE)

    if not os.path.exists(CHECKPOINT):
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    load_checkpoint_safely(model, CHECKPOINT, DEVICE)

    model.eval()

    correct = 0
    total = 0

    all_labels = []
    all_probs = []

    with torch.no_grad():

        for i, (trag_feat, clip_feat, labels, paths) in enumerate(tqdm(loader)):

            trag_feat = trag_feat.to(DEVICE)
            clip_feat = clip_feat.to(DEVICE)
            labels = labels.to(DEVICE)

            # 🔥 debug once
            if i == 0:
                print("[DEBUG] TRAG:", trag_feat.shape)
                print("[DEBUG] CLIP:", clip_feat.shape)

            logits = model(trag_feat, clip_feat)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # ===== METRICS =====
    accuracy = (correct / total * 100) if total > 0 else 0

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    roc_auc = roc_auc_score(all_labels, all_probs)

    all_preds = (all_probs >= 0.5).astype(int)

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)

    print("\n========== Intra Dataset Evaluation (SDFVD) ==========")
    print(f"Accuracy  : {accuracy:.2f}%")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("[DONE] SDFVD evaluation complete")


if __name__ == "__main__":
    main()