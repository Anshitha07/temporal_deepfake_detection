import torch
import torch.nn as nn


class FusionModel(nn.Module):
    def __init__(
        self,
        clip_dim=512,
        trag_dim=128,      # ✅ FIXED (matches your data)
        hidden_dim=128,
        num_classes=2
    ):
        super().__init__()

        # ------------------------------
        # CLIP Projection
        # ------------------------------
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ------------------------------
        # TRAG Projection (FIXED)
        # ------------------------------
        self.trag_proj = nn.Sequential(
            nn.Linear(trag_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # ------------------------------
        # Gating network
        # ------------------------------
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # ------------------------------
        # Classifier
        # ------------------------------
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, trag_feat, clip_feat, return_gate=False):

        # ---- Safety checks (VERY IMPORTANT) ----
        if trag_feat.shape[1] != 128:
            raise ValueError(f"Expected TRAG dim = 128, got {trag_feat.shape}")

        if clip_feat.shape[1] != 512:
            raise ValueError(f"Expected CLIP dim = 512, got {clip_feat.shape}")

        # ---- Project features ----
        clip_h = self.clip_proj(clip_feat)
        trag_h = self.trag_proj(trag_feat)

        # ---- Gate ----
        fusion_input = torch.cat([clip_h, trag_h], dim=1)
        gate = self.gate_net(fusion_input)

        # ---- Fusion ----
        fused = gate * clip_h + (1.0 - gate) * trag_h

        # ---- Classifier ----
        logits = self.classifier(fused)

        if return_gate:
            return logits, gate.squeeze(1)
        else:
            return logits