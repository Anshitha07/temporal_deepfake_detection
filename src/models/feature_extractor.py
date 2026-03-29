import torch
import torch.nn as nn
from torchvision.models import resnet18
from src.data.dataset import VideoFrameDataset


def main():
    # Load dataset
    dataset = VideoFrameDataset("data/processed/frames")
    frames, label = dataset[0]  # one video

    # frames: (T, C, H, W)
    print("Input frames shape:", frames.shape)

    # Load ResNet18 (pretrained)
    cnn = resnet18(pretrained=True)

    # Remove classification head
    cnn = nn.Sequential(*list(cnn.children())[:-1])  # output: (B, 512, 1, 1)

    # Freeze CNN
    for param in cnn.parameters():
        param.requires_grad = False

    cnn.eval()

    # Extract features per frame
    features = []
    with torch.no_grad():
        for t in range(frames.shape[0]):
            frame = frames[t].unsqueeze(0)  # (1, C, H, W)
            feat = cnn(frame)               # (1, 512, 1, 1)
            feat = feat.squeeze()           # (512,)
            features.append(feat)

    features = torch.stack(features)  # (T, 512)

    print("Extracted feature shape:", features.shape)
    print("Label:", label)


if __name__ == "__main__":
    main()
