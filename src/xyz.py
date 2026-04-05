import os
import numpy as np

root = "data/processed_ffpp/trag"

bad_files = []

for label in ["real", "fake"]:
    folder = os.path.join(root, label)

    for f in os.listdir(folder):
        if not f.endswith(".npy"):
            continue

        path = os.path.join(folder, f)

        try:
            arr = np.load(path)
            
            # Optional: check shape consistency
            if arr.ndim != 4 or arr.shape[1:] != (3, 224, 224):
                bad_files.append(path)

        except Exception as e:
            bad_files.append(path)

print("Bad files:", len(bad_files))

for f in bad_files[:10]:
    print(f)