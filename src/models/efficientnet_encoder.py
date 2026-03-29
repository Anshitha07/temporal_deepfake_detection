import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights


class EfficientNetEncoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()

        weights = EfficientNet_B4_Weights.IMAGENET1K_V1
        model = efficientnet_b4(weights=weights)

        # Remove classifier head
        self.backbone = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.device = device
        self.backbone.to(device)
        self.pool.to(device)

        for p in self.parameters():
            p.requires_grad = False

        self.eval()

    def forward(self, frames):
        """
        frames: (B, T, 3, 224, 224)
        returns: (B, 1280)
        """
        B, T, C, H, W = frames.shape
        frames = frames.view(B * T, C, H, W).to(self.device)

        with torch.no_grad():
            feats = self.backbone(frames)
            feats = self.pool(feats).squeeze(-1).squeeze(-1)

        feats = feats.view(B, T, -1)
        feats = feats.mean(dim=1)  # temporal pooling

        return feats
