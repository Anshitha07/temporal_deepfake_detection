import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

# ================= CONFIG =================
FEATURE_ROOT = "data/processed/efficientnet_features"
BATCH_SIZE = 32
EPOCHS = 10
LR = 1e-3
TRAIN_SPLIT = 0.8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =========================================


class FeatureDataset(Dataset):
    def __init__(self, root):
        self.samples = []

        for label_name, label in [("real", 0), ("fake", 1)]:
            d = os.path.join(root, label_name)
            for f in os.listdir(d):
                if f.endswith(".pt"):
                    self.samples.append((os.path.join(d, f), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return torch.load(path).float(), label


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1792, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.net(x)


def main():
    dataset = FeatureDataset(FEATURE_ROOT)
    train_size = int(TRAIN_SPLIT * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=BATCH_SIZE)

    model = Classifier().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        correct, total = 0, 0

        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                correct += (out.argmax(1) == y).sum().item()
                total += y.size(0)

        val_acc = correct / total

        print(f"Epoch [{epoch}/{EPOCHS}] | Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f}")

    print("[DONE] EfficientNet-only training complete")


if __name__ == "__main__":
    main()
