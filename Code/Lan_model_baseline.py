import torch, os, sys
import torch.nn as nn
import torchvision.transforms as T
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torchvision.models as models
from transformers import ViTModel, ViTConfig
import timm
import torch.nn.functional as F
from collections import OrderedDict


class VGG16(nn.Module):
    def __init__(self, pretrained=True, num_keypoints=4):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=pretrained)
        self.vgg16.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.vgg16.classifier[-1] = nn.Linear(self.vgg16.classifier[-1].in_features, num_keypoints * 2)

    def forward(self, x):
        x = self.vgg16(x)
        x = torch.sigmoid(x) * 224
        x = x.view(x.shape[0], -1, 2)
        return x

class ResNet18(nn.Module):
    def __init__(self, pretrained=True, num_keypoints=4):
        super(ResNet18, self).__init__()
        self.resnet = models.resnet18(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_keypoints * 2)

    def forward(self, x):
        x = self.resnet(x)  # (B, num_keypoints * 2)
        x = torch.sigmoid(x) * 224
        x = x.view(x.shape[0], -1, 2)  # Reshape to (B, num_keypoints, 2)
        return x

class ResNet50(nn.Module):
    def __init__(self, pretrained=True, num_keypoints=4):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=pretrained)
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_keypoints * 2)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.sigmoid(x) * 224
        x = x.view(x.shape[0], -1, 2)
        return x

class UNet(nn.Module):
    def __init__(self, pretrained=True, num_keypoints=4):
        super(UNet, self).__init__()
        self.unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                                   in_channels=3, out_channels=1, init_features=32, pretrained=True)
        # self.unet.encoder1.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) 
        self.fc = nn.Linear(224 * 224, num_keypoints * 2)  # Flatten output to keypoint predictions

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # [B,1,H,W] -> [B,3,H,W]
        x = self.unet(x)  # U-Net output (B, 1, 224, 224)
        x = x.view(x.shape[0], -1)  # Flatten (B, 50176)
        # print("*****", x.shape)
        x = self.fc(x)
        x = torch.sigmoid(x) * 224
        x = x.view(x.shape[0], -1, 2)  # Reshape to (B, num_keypoints, 2)
        return x

class ViT(nn.Module):
    def __init__(self, pretrained=True, num_keypoints=4):
        super(ViT, self).__init__()
        if pretrained:
            self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")
        else:
            config = ViTConfig()
            self.vit = ViTModel(config)
        
        self.fc = nn.Linear(self.vit.config.hidden_size, num_keypoints * 2)
    
    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # [B,1,H,W] -> [B,3,H,W]
        x = self.vit(x).last_hidden_state[:, 0, :]
        x = self.fc(x)
        x = torch.sigmoid(x) * 224
        x = x.view(x.shape[0], -1, 2)
        return x



#  VGG16, ResNet18, ResNet50, UNet, ViT
if __name__ == "__main__":
    model = UNet(pretrained=True, num_keypoints=4)
    dummy_input = torch.randn(2, 1, 224, 224)  # 1-channel
    coords_pred = model(dummy_input)
    print("Coords shape:", coords_pred.shape)    # [2,10] for 5 keypoints => 2D each
    # print("recover heatmap shape:", recover_heatmap.shape)    # [2,10] for 5 keypoints => 2D each
    print(coords_pred.max(), coords_pred.min())