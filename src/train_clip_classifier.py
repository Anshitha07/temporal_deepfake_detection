import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# ================= CONFIG =================
FEATURE_ROOT = "data/processed_ffpp/clip_features2"
BATCH_SIZE = 64
EPOCHS = 80
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
# =========================================


class CLIPFeatureDataset(Dataset):
    def __init__(self, feature_root):
        self.samples = []

        for label_name, label in [("real", 0), ("fake", 1)]:
            class_dir = os.path.join(feature_root, label_name)

            if not os.path.exists(class_dir):
                continue

            for f in os.listdir(class_dir):
                if f.endswith(".pt"):
                    self.samples.append((os.path.join(class_dir, f), label))

        print(f"[INFO] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        feature = torch.load(path).float()
        return feature, torch.tensor(label, dtype=torch.long)


class CLIPClassifier(nn.Module):
    def __init__(self, in_dim=1024):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2)
        )

    def forward(self, x):
        return self.net(x)


def normalize(x):
    return x / (x.norm(dim=1, keepdim=True) + 1e-6)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    correct, total = 0, 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        x = normalize(x)

        # 🔥 noise augmentation
        x = x + 0.01 * torch.randn_like(x)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)

        loss.backward()
        optimizer.step()

        correct += (out.argmax(dim=1) == y).sum().item()
        total += y.size(0)

    return correct / total


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)

            x = normalize(x)
            out = model(x)

            correct += (out.argmax(dim=1) == y).sum().item()
            total += y.size(0)

    return correct / total


def main():
    torch.manual_seed(SEED)

    dataset = CLIPFeatureDataset(FEATURE_ROOT)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = CLIPClassifier().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=3e-4,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS
    )

    best_val = 0
    patience = 10
    counter = 0

    for epoch in range(EPOCHS):

        train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_acc = evaluate(model, val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1} | Train: {train_acc:.4f} | Val: {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            counter = 0

            torch.save(model.state_dict(), "best_clip_model2.pth")

        else:
            counter += 1

        if counter >= patience:
            print("Early stopping triggered")
            break

    print(f"\nBest Validation Accuracy: {best_val:.4f}")


if __name__ == "__main__":
    main()