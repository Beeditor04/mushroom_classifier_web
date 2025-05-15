import torch
import torch.nn as nn
from torchvision import models

class ConvNeXt_Tiny(nn.Module):
    def __init__(self, num_classes=4, freeze=False, include_top=True, pretrained=True):
        super(ConvNeXt_Tiny, self).__init__()
        if pretrained:
            self.model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        else:
            self.model = models.convnext_tiny(weights=None)
        in_features = self.model.classifier[2].in_features
        self.model.classifier[2] = nn.Linear(in_features, num_classes)
        if not include_top:
            self.model.classifier[2] = nn.Identity()
        if freeze and pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.classifier.parameters():
                param.requires_grad = True
    def forward(self, x):
        return self.model(x)