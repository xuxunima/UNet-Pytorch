import torch.nn.functional as F
from .layers import *


class UNet(nn.Module):
    def __init__(self, in_channel, num_classes):
        super(UNet, self).__init__()
        self.conv1 = conv_block(in_channel, 64)
        self.down1 = down_layer(64, 128)
        self.down2 = down_layer(128, 256)
        self.down3 = down_layer(256, 512)
        self.down4 = down_layer(512, 1024)
        self.up1 = up_layer(1024, 512)
        self.up2 = up_layer(512, 256)
        self.up3 = up_layer(256, 128)
        self.up4 = up_layer(128, 64)
        self.out_layer = out_layer(64, num_classes)

    def forward(self, inputs):
        x1 = self.conv1(inputs)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out_layer(x)
        x = F.sigmoid(x)
        return x






