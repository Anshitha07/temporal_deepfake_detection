import os
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from src.data.trag_dataset import TRAGDataset
from src.models.trag_tcn import TRAG_TCN
from src.utils.trag_utils import compute_trag_input

# ================= CONFIG =================
DATASETS = [ "uadfv",  "ffpp_new"]

BASE_PATH = "data"
CHECKPOINT_DIR = "checkpoints"

BATCH_SIZE = 4
DEVICE = "cpu"
SEED = 42
# =========================================


def run_test(loader, model):

    correct, total = 0, 0

    with torch.no_grad():
        for x, y in tqdm(loader, leave=False):

            x = x.to(DEVICE)
            y = y.to(DEVICE)

            # TRAG input conversion fixed via utility helper
            x = compute_trag_input(x)

            out = model(x)
            pred = out.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    return correct / total


def get_test_dataset(dataset_name):

    trag_root = os.path.join(BASE_PATH, f"processed_{dataset_name}", "trag")

    # ------------------------------
    # 🔥 WDF → test
    # ------------------------------
    if dataset_name == "wdf":
        return TRAGDataset(trag_root, "test")

    # ------------------------------
    # 🔥 FFPP → val (no test)
    # ------------------------------
    if dataset_name == "ffpp_new":
        return TRAGDataset(trag_root, "val")

    # ------------------------------
    # 🔥 Others → recreate 80/20 split
    # ------------------------------
    full_dataset = TRAGDataset(trag_root)

    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    _, test_ds = random_split(
        full_dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    return test_ds


def test_dataset(dataset_name):

    print(f"\n🚀 Intra Testing: {dataset_name}")

    checkpoint = os.path.join(
        CHECKPOINT_DIR, f"trag_tcn_{dataset_name}_best.pth"
    )

    if not os.path.exists(checkpoint):
        print(f"❌ Missing checkpoint: {checkpoint}")
        return

    # ------------------------------
    # DATA
    # ------------------------------
    test_ds = get_test_dataset(dataset_name)
    loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    print(f"[INFO] Test samples: {len(test_ds)}")

    # ------------------------------
    # MODEL
    # ------------------------------
    model = TRAG_TCN().to(DEVICE)

    ckpt = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])

    model.eval()

    acc = run_test(loader, model)

    print(f"✅ {dataset_name} Intra Accuracy: {acc:.4f}")


def main():

    print("[INFO] Starting Intra Testing")

    for dataset_name in DATASETS:
        try:
            test_dataset(dataset_name)
        except Exception as e:
            print(f"❌ Error in {dataset_name}: {e}")


if __name__ == "__main__":
    main()