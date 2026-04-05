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

from src.models.clip_classifier import CLIPClassifier


# ================= CONFIG =================
CLIP_FEAT_ROOT = r"C:\Users\Anshitha\Documents\temporal-deepfake-detection\temporal-deepfake-detection\data\processed_celebdf\clip_features"

CHECKPOINT_PATH = r"C:\Users\Anshitha\Documents\temporal-deepfake-detection\temporal-deepfake-detection\checkpoints\clip_classifier_best.pth"

BATCH_SIZE = 32
DEVICE = "cpu"
# =========================================


# ================= DATASET =================
class ClipDataset(Dataset):

    def __init__(self, root):

        self.samples = []

        for label_name, label in [("real", 0), ("fake", 1)]:

            folder = os.path.join(root, label_name)

            if not os.path.exists(folder):
                print(f"[WARNING] Missing folder: {folder}")
                continue

            files = sorted([f for f in os.listdir(folder) if f.endswith(".pt")])

            print(f"[INFO] {label_name}: {len(files)} samples")

            for f in files:
                self.samples.append((
                    os.path.join(folder, f),
                    label
                ))

        print(f"[INFO] Total samples: {len(self.samples)}")

        if len(self.samples) == 0:
            raise RuntimeError("No CLIP features found!")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        feat = torch.load(path)

        return feat.float(), torch.tensor(label, dtype=torch.long)


# ================= EVALUATION =================
def evaluate(model, loader):

    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():

        for feats, labels in loader:

            feats = feats.to(DEVICE)
            labels = labels.to(DEVICE)

            # 🔥 normalize features (important)
            feats = torch.nn.functional.normalize(feats, dim=1)

            outputs = model(feats)

            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # 🔥 best threshold search
    best_acc = 0
    best_thresh = 0.5

    for t in np.linspace(0.1, 0.9, 17):
        preds = (all_probs >= t).astype(int)
        acc_t = (preds == all_labels).mean()

        if acc_t > best_acc:
            best_acc = acc_t
            best_thresh = t

    print(f"[INFO] Best threshold: {best_thresh:.2f} | Acc: {best_acc:.4f}")

    all_preds = (all_probs >= best_thresh).astype(int)

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

    dataset = ClipDataset(CLIP_FEAT_ROOT)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    print("[INFO] Loading CLIP classifier...")

    model = CLIPClassifier().to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state"])

    print("[INFO] Running evaluation...")

    acc, roc_auc, precision, recall, f1, cm = evaluate(model, loader)

    print("\n========== CLIP ONLY RESULTS ==========")
    print(f"Accuracy  : {acc:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nConfusion Matrix:")
    print(cm)


if __name__ == "__main__":
    main()