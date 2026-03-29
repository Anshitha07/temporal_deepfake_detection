import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else None
        )


    def forward(self, x):

        out = F.relu(self.bn1(self.conv1(x)))
        out = out[:, :, :x.size(2)]

        out = F.relu(self.bn2(self.conv2(out)))
        out = out[:, :, :x.size(2)]

        res = x if self.downsample is None else self.downsample(x)

        return out + res



class TRAG_TCN(nn.Module):

    def __init__(self, num_classes=2):
        super().__init__()

        self.tcn = nn.Sequential(
            TemporalBlock(4, 32, kernel_size=3, dilation=1),
            TemporalBlock(32, 64, kernel_size=3, dilation=2),
            TemporalBlock(64, 128, kernel_size=3, dilation=4),
        )

        self.classifier = nn.Linear(128, num_classes)


    def forward(self, x):
        """
        x : (B, T, 4)
        """

        x = x.transpose(1,2)   # (B,4,T)

        x = self.tcn(x)        # (B,128,T)

        x = x.mean(dim=2)      # (B,128)

        return self.classifier(x)