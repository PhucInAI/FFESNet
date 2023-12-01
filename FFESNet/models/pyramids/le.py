"""Simple Local Emphasis layer"""
from torch import nn
from torch.nn import functional as F

class LocalEmphasisLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dilation=1):
        super(LocalEmphasisLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding, dilation)
        self.attention = nn.Conv2d(in_channels, 1, kernel_size, padding, dilation)

    def forward(self, x):
        conv_out = self.conv(x)
        attention_out = self.attention(x)
        attention_out = F.sigmoid(attention_out)
        return conv_out * attention_out