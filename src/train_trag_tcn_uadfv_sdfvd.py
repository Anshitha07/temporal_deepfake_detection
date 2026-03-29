import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.trag_dataset import TRAGDataset
from src.models.trag_tcn import TRAG_TCN

# ================= CONFIG =================
DATASETS = ["uadfv", "sdfvd"]

BASE_PATH = "data"

BATCH_SIZE = 2          # 🔥 small for CPU
EPOCHS = 10
LR = 1e-4
NUM_WORKERS = 0         # 🔥 IMPORTANT for CPU
SEED = 42

DEVICE = "cpu"          # 🔥 FORCE CPU
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

        # TRAG input conversion
        x = x.mean(dim=(3, 4))
        zero = torch.zeros(B, T, 1)
        x = torch.cat((x, zero), dim=2)

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


def train_dataset(dataset_name):
    print(f"\n🚀 Training on {dataset_name.upper()} (CPU)")

    TRAG_ROOT = os.path.join(BASE_PATH, f"processed_{dataset_name}", "trag")

    dataset = TRAGDataset(TRAG_ROOT)

    if len(dataset) == 0:
        print(f"❌ No data for {dataset_name}")
        return

    # 🔥 create validation split
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    model = TRAG_TCN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):

        train_loss, train_acc = run_epoch(model, train_loader, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader)

        print(
            f"{dataset_name} | Epoch [{epoch}/{EPOCHS}] | "
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
                f"checkpoints/trag_tcn_{dataset_name}_best.pth"
            )

            print(f"✅ Saved BEST model ({dataset_name})")


def main():
    torch.manual_seed(SEED)

    print("[INFO] Running on CPU")

    for dataset_name in DATASETS:
        try:
            train_dataset(dataset_name)
        except Exception as e:
            print(f"❌ Error in {dataset_name}: {e}")


if __name__ == "__main__":
    main()