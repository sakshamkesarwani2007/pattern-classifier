"""The two models: a pretrained ResNet-18 and a small CNN trained from scratch.
"""
import torch
import torch.nn as nn
from torchvision import models

from groups import GROUP_NAMES

NUM_CLASSES = len(GROUP_NAMES)


def make_resnet(pretrained=True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model


def conv_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            conv_block(3, 16),
            conv_block(16, 32),
            conv_block(32, 64),
            conv_block(64, 128),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def get_model(name, pretrained=True):
    if name == "resnet":
        return make_resnet(pretrained)
    if name == "cnn":
        return SmallCNN()
    raise ValueError("model must be 'resnet' or 'cnn', got " + repr(name))


if __name__ == "__main__":
    x = torch.zeros(2, 3, 128, 128)
    for name in ["cnn", "resnet"]:
        model = get_model(name, pretrained=False)
        n_params = sum(p.numel() for p in model.parameters())
        print(name, "output shape:", tuple(model(x).shape), "parameters:", n_params)
