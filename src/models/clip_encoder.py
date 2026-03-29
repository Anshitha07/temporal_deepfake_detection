import torch
import torch.nn as nn
import torch.nn.functional as F
import clip


class CLIPVisualEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.clip_model.eval()
        self.clip_model.to(device)

        # Freeze CLIP
        for p in self.clip_model.parameters():
            p.requires_grad = False

        # CLIP normalization constants
        self.register_buffer(
            "mean",
            torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std",
            torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1)
        )

    def forward(self, frames):
        """
        frames: (B, T, 3, 224, 224) in [0,1]
        returns: (B, 512)
        """
        B, T, C, H, W = frames.shape

        frames = frames.view(B * T, C, H, W)
        frames = frames.float()

        # Normalize for CLIP
        frames = (frames - self.mean) / self.std
        frames = frames.to(self.device)

        with torch.no_grad():
            features = self.clip_model.encode_image(frames)

        features = features.view(B, T, -1)
        features = features.mean(dim=1)      # temporal pooling
        features = F.normalize(features, dim=-1)

        return features
