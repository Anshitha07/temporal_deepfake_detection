import os
import torch
import cv2
import numpy as np

from src.models.fusion_model import FusionModel

# =========================
# CONFIG
# =========================
MODEL_PATH = "checkpoints/fusion_best_celebdf.pth"

TRAG_ROOT = "data/processed_celebdf/trag_features"
CLIP_ROOT = "data/processed_celebdf/clip_features"
FRAMES_ROOT = "data/processed_celebdf/frames"

OUTPUT_DIR = "results/visualizations"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================
# LOAD MODEL (SAFE)
# =========================
def load_model():
    print("[INFO] Loading model...")

    model = FusionModel().to(DEVICE)
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)

    if "fusion_model" in ckpt:
        ckpt = ckpt["fusion_model"]
    elif "model_state" in ckpt:
        ckpt = ckpt["model_state"]
    elif "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # remove "module." if present
    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith("module."):
            k = k[7:]
        new_ckpt[k] = v

    model_dict = model.state_dict()
    filtered_ckpt = {}

    for k, v in new_ckpt.items():
        if k in model_dict and model_dict[k].shape == v.shape:
            filtered_ckpt[k] = v

    model_dict.update(filtered_ckpt)
    model.load_state_dict(model_dict)

    print(f"[INFO] Loaded {len(filtered_ckpt)} layers\n")

    model.eval()
    return model


# =========================
# VISUALIZATION FUNCTION
# =========================
def visualize_samples(model, num_samples=5):

    print("[INFO] Generating visualizations...\n")

    for label in ["real", "fake"]:

        trag_dir = os.path.join(TRAG_ROOT, label)
        clip_dir = os.path.join(CLIP_ROOT, label)
        frame_dir = os.path.join(FRAMES_ROOT, label)

        videos = os.listdir(trag_dir)[:num_samples]

        for vid_file in videos:

            if not vid_file.endswith(".pt"):
                continue

            video_name = vid_file.replace(".pt", "")

            trag_path = os.path.join(trag_dir, vid_file)
            clip_path = os.path.join(clip_dir, vid_file)

            frame_folder = os.path.join(frame_dir, video_name)

            if not os.path.exists(clip_path) or not os.path.exists(frame_folder):
                continue

            # 🔥 Load features
            trag_feat = torch.load(trag_path).unsqueeze(0).to(DEVICE)
            clip_feat = torch.load(clip_path).unsqueeze(0).to(DEVICE)

            # 🔥 Get one frame
            frames = sorted(os.listdir(frame_folder))
            if len(frames) == 0:
                continue

            frame_path = os.path.join(frame_folder, frames[0])

            img = cv2.imread(frame_path)
            img = cv2.resize(img, (224, 224))

            # 🔥 Forward pass
            with torch.no_grad():
                logits, gate = model(trag_feat, clip_feat, return_gate=True)

                probs = torch.softmax(logits, dim=1)
                confidence = probs[0, 1].item()
                pred = torch.argmax(probs, dim=1).item()

            pred_label = "FAKE" if pred == 1 else "REAL"

            # =========================
            # DRAW RESULTS
            # =========================
            overlay = img.copy()

            color = (0, 0, 255) if pred == 1 else (0, 255, 0)

            cv2.putText(
                overlay,
                f"Pred: {pred_label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

            cv2.putText(
                overlay,
                f"Conf: {confidence:.2f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                overlay,
                f"Gate: {gate.item():.2f}",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),
                2
            )

            # Save
            save_path = os.path.join(
                OUTPUT_DIR, f"{label}_{video_name}.png"
            )

            cv2.imwrite(save_path, overlay)

            print(f"[SAVED] {save_path}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    model = load_model()
    visualize_samples(model)

    print("\n[DONE] Visualization complete")