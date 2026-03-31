import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

os.environ["TORCHDYNAMO_DISABLE"] = "1"

from src.data.trag_dataset import TRAGDataset
from src.models.trag_tcn import TRAG_TCN

# ================= CONFIG =================
TRAG_ROOT = "data/processed_ffpp/trag"

BATCH_SIZE = 2   # 🔥 safer for BIG dataset
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 2
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================

os.makedirs("checkpoints", exist_ok=True)


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    criterion = nn.CrossEntropyLoss()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in tqdm(loader, leave=False):

        x = x.to(DEVICE)
        y = y.to(DEVICE)

        B, T, C, H, W = x.shape

        # 🔥 spatial collapse
        x = x.mean(dim=(3, 4))   # (B, T, C)

        zero = torch.zeros(B, T, 1, device=x.device)
        x = torch.cat((x, zero), dim=2)  # (B, T, 4)

        with torch.set_grad_enabled(is_train):
            out = model(x)
            loss = criterion(out, y)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * x.size(0)
        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total


def main():
    torch.manual_seed(SEED)

    print("[INFO] Device:", DEVICE)

    dataset = TRAGDataset(TRAG_ROOT)

    print(f"[INFO] Total samples: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    print(f"[INFO] Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = TRAG_TCN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader)

        print(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc
                },
                "checkpoints/trag_tcn_best.pth"
            )

            print("✅ Saved BEST model")

    print("[DONE]")


if __name__ == "__main__":
    main()