import torch
import torch.nn as nn
import torchvision

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

__all__ = ['ResNet50', 'ResNet101', 'ResNet152']  # __all__属性，可以用于**模块导入时限制**


class Bottleneck(nn.Module):
    def __init__(self, in_channels, filters, stride=1, is_1x1conv=False, expansion=4):
        super(Bottleneck, self).__init__()
        self.expansion = expansion
        self.is_1x1conv = is_1x1conv

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=filters, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filters, out_channels=filters, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=filters, out_channels=filters * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(filters * self.expansion)
        )

        if self.is_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=filters * self.expansion, kernel_size=1, stride=1,
                          bias=False),
                nn.BatchNorm2d(filters * self.expansion)
            )

    def forward(self, x):
        x_shortcut = x
        x = self.bottleneck(x)

        if self.is_1x1conv:
            x_shortcut = self.shortcut(x_shortcut)

        out = x + x_shortcut
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, blocks, num_classes, num_channels=3, expansion=4):
        super(ResNet, self).__init__()
        self.expansion = expansion

        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(in_channels=64, filters=64, block=blocks[0], stride=1)
        self.layer2 = self._make_layer(in_channels=256, filters=128, block=blocks[1], stride=2)
        self.layer3 = self._make_layer(in_channels=512, filters=256, block=blocks[2], stride=2)
        self.layer4 = self._make_layer(in_channels=1024, filters=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, filters, block, stride):
        layers = []
        layers.append(Bottleneck(in_channels, filters, stride, is_1x1conv=True))
        for i in range(1, block):
            layers.append(Bottleneck(filters * self.expansion, filters, stride=1, is_1x1conv=False))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def ResNet50(num_classes, num_channels):
    return ResNet([3, 4, 6, 3], num_classes=num_classes, num_channels=num_channels)


def ResNet101(num_classes, num_channels):
    return ResNet([3, 4, 23, 3], num_classes=num_classes, num_channels=num_channels)


def ResNet152(num_classes, num_channels):
    return ResNet([3, 8, 36, 3], num_classes=num_classes, num_channels=num_channels)
