import os
import shutil

src = "data/processed_wdf/frames/test/real"
dst = "data/processed_wdf/frames_grouped/test/real"

os.makedirs(dst, exist_ok=True)

for f in os.listdir(src):

    if not f.endswith(".png"):
        continue

    vid = f.split("_")[0]

    video_folder = os.path.join(dst, vid)
    os.makedirs(video_folder, exist_ok=True)

    shutil.move(
        os.path.join(src, f),
        os.path.join(video_folder, f)
    )