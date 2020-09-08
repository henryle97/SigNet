from torch import nn
import torch


class ContrastiveLoss(nn.Module):
    def __init__(self, alpha, beta, margin=1):
        self.alpha = alpha
        self.beta = beta
        self.margin = margin

    def forward(self, s1, s2, y):

        euclid_distance = torch.pairwise_distance(s1, s2, p=2)

        loss = self.alpha * (1-y) * euclid_distance**2 + self.beta * y * torch.max(torch.zeros_like(euclid_distance), self.margin - euclid_distance**2)

        return torch.mean(loss, dtype=torch.float)

