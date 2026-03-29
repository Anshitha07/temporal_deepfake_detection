# Temporal Deepfake Detection (CelebDF pipeline)

This repository implements a temporal deepfake detection pipeline for CelebDF-like datasets, with support for:

- face extraction from video frames
- TRAG feature generation and TRAG-TCN model training
- CLIP visual feature extraction
- fusion of TRAG + CLIP features with classification
- cross/intra evaluation workflow

## 🚀 Quick Setup

1. Create and activate venv:
```powershell
python -m venv venv
.\\venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. Prepare input videos (real/fake structure):

```
<root>/data/celeb_df_small/real/*.mp4
<root>/data/celeb_df_small/fake/*.mp4
```

3. Run the preparation and feature pipeline:

```powershell
python src/prepare_celebdf.py
python src/extract_faces_celebdf.py
python src/extract_trag_input.py
python src/extract_trag_features.py
python src/extract_clip_features.py
python src/train_fusion.py
python src/test_fusion.py
python src/test_fusion_intra.py
```

## 🛠️ Files Added/Updated

- `src/data/face_preprocessing.py` : face detection helper
- `src/data/clip_dataset.py` : CLIP dataset loader
- `src/extract_faces_celebdf.py` : face crop step from frames
- `src/extract_trag_input.py` : frame->TRAG .npy conversion
- `src/extract_trag_features.py` : TRAG feature extraction
- `src/extract_clip_features.py` : CLIP feature extraction
- `src/train_fusion.py`, `src/test_fusion.py`, `src/test_fusion_intra.py` : fusion train/test

## 📦 Directory layout

- `data/processed_celebdf/frames/real|fake/<video_id>/*.jpg`
- `data/processed_celebdf/faces/real|fake/<video_id>/*.jpg`
- `data/processed_celebdf/trag/real|fake/<video_id>.npy`
- `data/processed_celebdf/trag_features/real|fake/<video_id>.pt`
- `data/processed_celebdf/clip_features/real|fake/<video_id>.pt`

## 💡 Notes

- If `ModuleNotFoundError: No module named 'src'`, run scripts from repo root (corrected in scripts by adding `sys.path` entry).
- `prepare_celebdf.py` does 80/20 split by default.
- For GPU set `DEVICE = "cuda"` in scripts.

## 🧹 Add .gitignore (recommended)

```
venv/
__pycache__/
*.pyc
data/processed_celebdf/
checkpoints/
results/
```
