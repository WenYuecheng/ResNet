import torch
import torch.nn as nn
from torchsummary import summary


class Residual(nn.Module):
    def __init__(self, input_channel, output_channel, use_1conv=False, stride=1):
        super(Residual, self).__init__()
        self.ReLu = nn.ReLU()
        self.Conv1 = nn.Conv2d(
            input_channel, output_channel, kernel_size=3, padding=1, stride=stride
        )
        self.Conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.bn2 = nn.BatchNorm2d(output_channel)
        if use_1conv:
            self.Conv3 = nn.Conv2d(
                input_channel, output_channel, kernel_size=1, stride=stride
            )
        else:
            self.Conv3 = None

    def forward(self, x):
        y = self.ReLu(self.bn1(self.Conv1(x)))
        y = self.bn2(self.Conv2(y))
        if self.Conv3:
            x = self.Conv3(x)
        return self.ReLu(x + y)

class ResNet(nn.Module):
    def __init__(self, Residual):
        super(ResNet, self).__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        self.b2 = nn.Sequential(
            Residual(64, 64), Residual(64, 64)
        )
        self.b3 = nn.Sequential(
            Residual(64, 128, use_1conv=True, stride=2), Residual(128, 128)
        )
        self.b4 = nn.Sequential(
            Residual(128, 256, use_1conv=True, stride=2), Residual(256, 256)
        )
        self.b5 = nn.Sequential(
            Residual(256, 512, use_1conv=True, stride=2), Residual(512, 512)
        )
        self.b6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x