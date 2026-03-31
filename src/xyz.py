import os

trag = os.listdir("data/processed_ffpp/trag_features/real")
clip = os.listdir("data/processed_ffpp/clip_features/real")

print("TRAG sample:", trag[:5])
print("CLIP sample:", clip[:5])

print("INTERSECTION:", len(set(trag) & set(clip)))