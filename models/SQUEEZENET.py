import torch
import torch.nn as nn

class FireModule(nn.Module):
    def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
        super().__init__()
        self.squeeze= nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
        self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
        self.expand_activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return self.expand_activation(torch.cat([
            self.expand1x1(x),
            self.expand3x3(x)
        ], 1))
    
class SqueezeNet(nn.Module):
    def __init__(self, num_classes=1000, num_fire_modules=8,
                 base_e=128, incre=128, freq=2, pct3x3=0.5, squeeze_ratio=0.125):
        super().__init__()
        self.setm=nn.Sequential(
            nn.Conv2d(3,96,kernel_size=7,stride=2),
            nn.ReLU(inplace=True),
        )

        self.fire_modules=nn.Sequential()
        in_channels=96

        for i in range(num_fire_modules):
            if i in {0,3,7}:
                self.setm.add_module(f"maxpool_{i+1}", nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))
            e_i = base_e + (i // freq) * incre
            e_3x3 = int(e_i * pct3x3)
            e_1x1 = e_i - e_3x3
            s_channels = int(squeeze_ratio * (e_i))

            fire_module = FireModule(in_channels, s_channels, e_1x1, e_3x3)
            self.fire_modules.add_module(f"fire_{i+2}", fire_module)
            in_channels = e_i

        self.final_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            self.final_conv,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight,nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.setm(x)
        x = self.fire_modules(x)
        x = self.classifier(x)
        return torch.flatten(x, 1)