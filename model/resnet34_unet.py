import torch
import torch.nn as nn
import torch.nn.functional as F
from thop import profile

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.double_conv(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet34Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(ResNet34Encoder, self).__init__()
        self.in_channels = 64

        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, self.in_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.in_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * BasicBlock.expansion
        for _ in range(1, blocks):
            layers.append(BasicBlock(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.inc(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        return x4
    
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Decoder(nn.Module):
    def __init__(self, in_channels ,out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels , in_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.up(x)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetPlusResNet34(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNetPlusResNet34, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.encoder = ResNet34Encoder()
    
        self.inc = self.encoder.inc
        self.encoder1 = self.encoder.layer1
        self.encoder2 = self.encoder.layer2
        self.encoder3 = self.encoder.layer3
        self.encoder4 = self.encoder.layer4

        self.middle = DoubleConv(512, 256)

        self.decoder1 = Decoder(768, 32)
        self.decoder2 = Decoder(288, 32)
        self.decoder3 = Decoder(160, 32)
        self.decoder4 = Decoder(96, 32)
        self.decoder5 = nn.Sequential(
            nn.ConvTranspose2d(32 , 32, kernel_size=2, stride=2),
            DoubleConv(32, 32)
        )
        self.outc = OutConv(32, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)
        x6 = self.middle(x5)
        x = self.decoder1(x6, x5)
        x = self.decoder2(x, x4)
        x = self.decoder3(x, x3)
        x = self.decoder4(x, x2)
        x = self.decoder5(x)
        out = self.outc(x)
        return out
     
if __name__ == "__main__":
    # input_tensor = torch.rand((1, 3, 224, 224))
    # model = UNetPlusResNet34(n_channels=3, n_classes=1)
    # with torch.no_grad():
    #     output = model(input_tensor)

    # print(output.shape)

    net = UNetPlusResNet34(n_channels=3, n_classes=1).cuda()
    input = torch.randn(1, 3, 256, 256).cuda()
    flops, params = profile(net, (input, ))
    print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    print('Params = ' + str(params / 1000 ** 2) + 'M')
