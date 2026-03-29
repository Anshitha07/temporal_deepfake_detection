import torch
from clip_encoder import CLIPVisualEncoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Fake batch: B=2 videos, T=16 frames
x = torch.rand(2, 16, 3, 224, 224).to(DEVICE)

model = CLIPVisualEncoder(device=DEVICE).to(DEVICE)

out = model(x)

print("CLIP output shape:", out.shape)
print("Device:", out.device)
print("Dtype:", out.dtype)
