import torch.nn as nn
import torch

class conv_block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, 1, padding=1),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True),
                                  nn.Conv2d(out_ch, out_ch, 3, 1, padding=1),
                                  nn.BatchNorm2d(out_ch),
                                  nn.ReLU(inplace=True)
                                  )


    def forward(self, inputs):
        x = self.conv(inputs)
        return x


class down_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down_layer, self).__init__()
        self.down_conv = nn.Sequential(nn.MaxPool2d(kernel_size=2),
                                       conv_block(in_ch, out_ch))

    def forward(self, inputs):
        x = self.down_conv(inputs)
        return x


class up_layer(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(up_layer, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, padding=1)
        self.block = conv_block(in_ch, out_ch)

    def forward(self, inputs, prev):
        x = self.up(inputs)
        x = self.conv(x)

        x = torch.cat([prev, x], dim=1)
        x = self.block(x)
        return x


class out_layer(nn.Module):
    def __init__(self, in_ch, num_classes):
        super(out_layer, self).__init__()
        self.conv = nn.Conv2d(in_ch, num_classes, 1)

    def forward(self, inputs):
        x = self.conv(inputs)
        return x

