import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from src.data.subject_split_utils import subject_disjoint_split

# ================= CONFIG =================
FEATURE_ROOT = "data/processed_ffpp_new/clip_features"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
TRAIN_SPLIT = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
# =========================================


class CLIPFeatureDataset(Dataset):
    def __init__(self, feature_root):
        self.samples = []

        split_dirs = []

        train_dir = os.path.join(feature_root, "train")
        val_dir = os.path.join(feature_root, "val")

        if os.path.exists(train_dir):
            split_dirs.append(train_dir)

        if os.path.exists(val_dir):
            split_dirs.append(val_dir)

        if len(split_dirs) == 0:
            split_dirs.append(feature_root)

        for split_dir in split_dirs:

            for label_name, label in [("real", 0), ("fake", 1)]:
                class_dir = os.path.join(split_dir, label_name)

                if not os.path.exists(class_dir):
                    continue

                for f in sorted(os.listdir(class_dir)):
                    if f.endswith(".pt"):
                        video_name = f.replace(".pt", "")
                        subject_id = self._get_subject_id(video_name)

                        self.samples.append(
                            (os.path.join(class_dir, f), label, subject_id)
                        )

        print(f"[INFO] Loaded {len(self.samples)} CLIP feature samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, _ = self.samples[idx]
        feature = torch.load(path).float()
        label = torch.tensor(label, dtype=torch.long)
        return feature, label

    def _get_subject_id(self, video_name):
        return video_name.split("_")[0]

    def create_subset(self, samples):
        subset = self.__class__.__new__(self.__class__)
        subset.__dict__ = self.__dict__.copy()
        subset.samples = samples
        return subset


class CLIPClassifier(nn.Module):
    def __init__(self, in_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    correct, total = 0, 0

    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return correct / total


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


def main():
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    dataset = CLIPFeatureDataset(FEATURE_ROOT)

    train_samples, val_samples = subject_disjoint_split(
        dataset.samples,
        train_ratio=TRAIN_SPLIT,
        seed=SEED
    )

    train_set = dataset.create_subset(train_samples)
    val_set = dataset.create_subset(val_samples)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        pin_memory=True
    )

    model = CLIPClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    os.makedirs("checkpoints", exist_ok=True)
    best_val_acc = 0.0

    print("[INFO] Training CLIP-only classifier (subject-disjoint)")

    for epoch in range(1, EPOCHS + 1):

        train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc = evaluate(model, val_loader)

        print(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}"
        )

        # ✅ SAVE BEST MODEL
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc
                },
                "checkpoints/clip_classifier_best.pth"
            )

            print(f"[INFO] Saved best CLIP model (Val Acc = {val_acc:.4f})")

    print("[DONE] CLIP-only training finished")


if __name__ == "__main__":
    main()