import torch
import torch.nn as nn
from torchvision import models

class TinyViT(nn.Module):
    def __init__(self, num_classes=4, freeze=False, include_top=True, pretrained=True):
        super(TinyViT, self).__init__()
        self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
        in_features = self.model.heads[0].in_features
        self.model.heads[0] = nn.Linear(in_features, num_classes)
        if not include_top:
            self.model.heads[0] = nn.Identity()
        if freeze and pretrained:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.heads.parameters():
                param.requires_grad = True
        
    def forward(self, x):
        return self.model(x)