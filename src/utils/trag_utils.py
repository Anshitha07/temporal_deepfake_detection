import torch


def compute_trag_input(x):
    """Convert video tensor to TRAG input format: (B, T, C, H, W) -> (B, T, 4)."""
    B, T, C, H, W = x.shape
    x = x.mean(dim=(3, 4))           # (B, T, C)
    zero = torch.zeros(B, T, 1, device=x.device, dtype=x.dtype)
    x = torch.cat((x, zero), dim=-1) # (B, T, C+1)
    return x
