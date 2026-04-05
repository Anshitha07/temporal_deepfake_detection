import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

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
TRAG_FEAT_ROOT   = "data/processed_ffpp/trag_features"
CLIP_FEAT_ROOT   = "data/processed_ffpp/clip_features"

BATCH_SIZE       = 8
EPOCHS           = 50
LR               = 5e-4
DEVICE           = "cpu"
SEED             = 42

CHECKPOINT_PATH  = "checkpoints/fusion_best_ffpp.pth"
VAL_INDICES_PATH = "checkpoints/fusion_val_indices_ffpp.json"
# =========================================


class FusionDataset(Dataset):

    def __init__(self, trag_root, clip_root):
        self.samples = []

        for label_name, label in [("real", 0), ("fake", 1)]:
            trag_dir = os.path.join(trag_root, label_name)
            clip_dir = os.path.join(clip_root, label_name)

            if not os.path.exists(trag_dir) or not os.path.exists(clip_dir):
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
        trag_feat = torch.load(trag_path, weights_only=False)
        clip_feat = torch.load(clip_path, weights_only=False)
        return (
            trag_feat.float(),
            clip_feat.float(),
            torch.tensor(label, dtype=torch.long)
        )


# ================= TRAIN =================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    correct, total = 0, 0

    for trag_feat, clip_feat, labels in loader:
        trag_feat = trag_feat.to(DEVICE)
        clip_feat = clip_feat.to(DEVICE)
        labels    = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(trag_feat, clip_feat)
        loss    = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total   += labels.size(0)

    return correct / total


# ================= EVAL =================
def evaluate(model, loader):
    model.eval()

    all_labels, all_probs, all_gates = [], [], []
    correct, total = 0, 0

    with torch.no_grad():
        for trag_feat, clip_feat, labels in loader:
            trag_feat = trag_feat.to(DEVICE)
            clip_feat = clip_feat.to(DEVICE)
            labels    = labels.to(DEVICE)

            outputs, gates = model(trag_feat, clip_feat, return_gate=True)
            probs  = torch.softmax(outputs, dim=1)[:, 1]

            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total   += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_gates.extend(gates.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    all_preds  = (all_probs >= 0.5).astype(int)

    acc       = correct / total
    roc_auc   = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds)
    recall    = recall_score(all_labels, all_preds)
    f1        = f1_score(all_labels, all_preds)
    cm        = confusion_matrix(all_labels, all_preds)

    os.makedirs("results", exist_ok=True)
    np.save("results/fusion_gates.npy",  all_gates)
    np.save("results/fusion_labels.npy", all_labels)

    return acc, roc_auc, precision, recall, f1, cm


# ================= MAIN =================
def main():
    torch.manual_seed(SEED)
    print("[INFO] Using device:", DEVICE)

    full_dataset = FusionDataset(TRAG_FEAT_ROOT, CLIP_FEAT_ROOT)

    train_size = int(0.8 * len(full_dataset))
    val_size   = len(full_dataset) - train_size

    train_set, val_set = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )
    print(f"[INFO] Train: {len(train_set)} | Val: {len(val_set)}")

    os.makedirs("checkpoints", exist_ok=True)
    val_indices = val_set.indices if hasattr(val_set, 'indices') else list(range(train_size, len(full_dataset)))
    with open(VAL_INDICES_PATH, 'w') as f:
        import json
        json.dump(val_indices, f)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    model     = FusionModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", patience=5, factor=0.5
    )

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc, roc_auc, precision, recall, f1, cm = evaluate(model, val_loader)

        scheduler.step(val_acc)

        print(f"Epoch [{epoch}/{EPOCHS}] | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"ROC-AUC: {roc_auc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "fusion_model": model.state_dict(),
                "epoch":        epoch,
                "val_acc":      val_acc
            }, CHECKPOINT_PATH)
            print(f"[INFO] Saved best model (Val Acc = {val_acc:.4f})")

    print("\n========== Final Results ==========")
    print(f"Accuracy  : {val_acc:.4f}")
    print(f"ROC-AUC   : {roc_auc:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("[DONE] Training complete")


if __name__ == "__main__":
    main()