import torch
import torch.nn as nn
import torch.nn.functional as F

class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, input, target):
        # Compute the binary cross-entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        # Compute the focal loss
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss

        return focal_loss
