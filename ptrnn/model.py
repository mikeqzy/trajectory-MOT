import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvRNN(nn.Module):
    def __init__(self, C=32):
        super(ConvRNN, self).__init__()
        self.w_c = nn.Conv2d(2 * C, C, 3, padding=1, bias=False)
        self.gn = nn.GroupNorm(C // 16, C)
        self.relu = nn.ReLU(inplace=True)
        self.w_w = nn.Conv2d(C, C, 3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, prev, x):
        x = torch.cat([prev, x], dim=1) # [B, C, H, W]
        x = self.sigmoid(self.w_w(self.relu(self.gn(self.w_c(x)))))
        # x = self.sigmoid(self.w_w(self.relu(self.w_c(x))))
        return x