import torch
import torch.nn as nn

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels, shortcut=None):
        super().__init__()
        assert shortcut in (None, "simple", "complex"), \
            "shortcut must be None, 'simple', or 'complex'"
        self.shortcut = shortcut
        self.squeeze= nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand_activation = nn.ReLU(inplace=True)
        self.proj = None
        if shortcut == "complex" :
            self.proj = nn.Conv2d(in_channels, expand1x1_channels+expand3x3_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        x = self.squeeze_activation(self.squeeze(x))
        x=torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1)
        if self.shortcut == "simple":
            if residual.shape[1] == x.shape[1]:
                x += residual
        elif self.shortcut == "complex":
            if self.proj is not None:
                residual = self.proj(residual)
            x += residual
        return self.expand_activation(x)