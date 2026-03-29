import os
import torch

CKPT_DIR = "checkpoints"

print("Epoch-wise Validation Accuracies:\n")

ckpts = sorted(os.listdir(CKPT_DIR))

for ckpt in ckpts:
    if ckpt.endswith(".pth"):
        path = os.path.join(CKPT_DIR, ckpt)
        data = torch.load(path, map_location="cpu")

        epoch = data.get("epoch", "N/A")
        val_acc = data.get("val_acc", "N/A")

        print(f"{ckpt:25s} | Epoch: {epoch} | Val Acc: {val_acc:.4f}")
