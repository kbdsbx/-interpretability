import torch
import torch.nn as nn
import torchvision

# print("Pytorch version", torch.__version__)
# print("Torchvision version", torchvision.__version__)

class Bottleneck(nn.Module) :
    def __init__(self, in_channels, channels, stride = 1, downsampling = False, expansion = 4):
        super(Bottleneck, self).__init__()

        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels=channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = channels, out_channels=channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = channels, out_channels=channels * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels * self.expansion),
        )

        if self.downsampling :
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * self.expansion)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward (self, x) :
        bypass = x
        out = self.bottleneck(x)

        if self.downsampling :
            bypass = self.downsample(bypass)
        
        out += bypass
        out = self.relu(out)

        return out

class ResNet(nn.Module) :
    """
    Input : ? x 3 x 224 x 224
    Output : 1 x 1000
    """
    def __init__(self, blocks, num_classes = 1000, expansion = 4) :
        super(ResNet, self).__init__()

        self.expansion = expansion

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels= 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self.make_layer(in_clannels=64, channels=64, block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_clannels=256, channels=128, block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_clannels=512, channels=256, block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_clannels=1024, channels=512, block=blocks[3], stride=2)

        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules() :
            if isinstance(m, nn.Conv2d) :
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) :
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_clannels, channels, block, stride) :
        layers = []
        layers.append(Bottleneck(in_clannels, channels, stride, downsampling=True))
        for i in range(1, block) :
            layers.append(Bottleneck(channels * self.expansion, channels))
        
        return nn.Sequential(*layers)

    def forward(self, x) :
        x = self.conv1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def ResNet50() :
    return ResNet([3, 4, 6, 3])

def ResNet101() :
    return ResNet([3, 4, 23, 3])

def ResNet152() :
    return ResNet([3, 8, 36, 3])