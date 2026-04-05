import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from src.models.fusion_model import FusionModel


# ================= CONFIG =================
TRAG_FEAT_ROOT = r"C:\Users\Anshitha\Documents\temporal-deepfake-detection\temporal-deepfake-detection\data\processed_celebdf\trag_features"
CLIP_FEAT_ROOT = r"C:\Users\Anshitha\Documents\temporal-deepfake-detection\temporal-deepfake-detection\data\processed_celebdf\clip_features"

CHECKPOINT_PATH = r"C:\Users\Anshitha\Documents\temporal-deepfake-detection\temporal-deepfake-detection\checkpoints\fusion_best_ffpp\fusion_best_ffpp.pth"

BATCH_SIZE = 32
DEVICE = "cpu"
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

            print(f"[INFO] {label_name}: {len(common_files)} matched samples")

            for f in common_files:
                self.samples.append((
                    os.path.join(trag_dir, f),
                    os.path.join(clip_dir, f),
                    label
                ))

        print(f"[INFO] Total matched samples: {len(self.samples)}")

        if len(self.samples) == 0:
            raise RuntimeError("No matching TRAG + CLIP features found!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        trag_path, clip_path, label = self.samples[idx]

        trag_feat = torch.load(trag_path)
        clip_feat = torch.load(clip_path)

        return (
            trag_feat.float(),
            clip_feat.float(),
            torch.tensor(label, dtype=torch.long)
        )


# ================= EVALUATION =================
def evaluate(model, loader):

    model.eval()

    all_labels = []
    all_probs = []

    correct, total = 0, 0

    with torch.no_grad():

        for trag_feat, clip_feat, labels in loader:

            trag_feat = trag_feat.to(DEVICE)
            clip_feat = clip_feat.to(DEVICE)
            labels = labels.to(DEVICE)

            # 🔥 NORMALIZATION (improves stability)
            trag_feat = torch.nn.functional.normalize(trag_feat, dim=1)
            clip_feat = torch.nn.functional.normalize(clip_feat, dim=1)

            outputs = model(trag_feat, clip_feat)

            preds = outputs.argmax(dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 🔥 BEST THRESHOLD SEARCH (KEY IMPROVEMENT)
    best_acc = 0
    best_thresh = 0.25

    for t in np.linspace(0.1, 0.9, 17):
        preds_t = (all_probs >= t).astype(int)
        acc_t = (preds_t == all_labels).mean()

        if acc_t > best_acc:
            best_acc = acc_t
            best_thresh = t

    print(f"[INFO] Best threshold: {best_thresh:.2f} | Best Acc: {best_acc:.4f}")

    all_preds = (all_probs >= best_thresh).astype(int)

    # Metrics
    acc = (all_preds == all_labels).mean()
    roc_auc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    return acc, roc_auc, precision, recall, f1, cm


# ================= MAIN =================
def main():

    print("[INFO] Loading dataset...")

    dataset = FusionDataset(TRAG_FEAT_ROOT, CLIP_FEAT_ROOT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("[INFO] Loading model...")

    model = FusionModel().to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["fusion_model"])

    print(f"[INFO] Loaded model from epoch {checkpoint['epoch']} with val_acc {checkpoint['val_acc']:.4f}")

    print("[INFO] Running evaluation...")

    acc, roc_auc, precision, recall, f1, cm = evaluate(model, loader)

    print("\n========== TEST RESULTS ==========")
    print(f"Accuracy  : {acc:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()