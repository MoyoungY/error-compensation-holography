import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self, lambda_feat: float = 0.025):
        super().__init__()
        self.lambda_feat = lambda_feat
        self.vgg = self._setup_vgg()
        self.mse = nn.MSELoss()
        
    def _setup_vgg(self):
        """Setup VGG model for perceptual loss"""
        vgg = models.vgg16(pretrained=True)
        return nn.Sequential(*list(vgg.features)[:16]).eval()
        
    def forward(self, x, target):
        feat_x = self.vgg(x)
        feat_target = self.vgg(target)
        return self.lambda_feat * self.mse(feat_x, feat_target) 