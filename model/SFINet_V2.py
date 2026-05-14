from torch import nn
from torch import Tensor
import torch
import torchvision.models as models
import torch.nn.functional as F
from model.MobileNetV2 import mobilenet_v2


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Reduction(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Reduction, self).__init__()
        self.reduce = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, 3, padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=1)
        )

    def forward(self, x):

        return self.reduce(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 2, 1, bias=False),
                                nn.ReLU(),
                                nn.Conv2d(in_planes // 2, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x1 = torch.cat([avg_out, max_out], dim=1)
        x2 = self.conv1(x1)
        return self.sigmoid(x2)

class TransBasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=2, stride=2, padding=0, dilation=1, bias=False):
        super(TransBasicConv2d, self).__init__()
        self.Deconv = nn.ConvTranspose2d(in_planes, out_planes,
                                         kernel_size=kernel_size, stride=stride,
                                         padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.Deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class MAAF(nn.Module):
    def __init__(self, channel):
        super(MAAF, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(channel, channel, kernel_size=(3, 1), padding=(1, 0))
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(channel, channel, kernel_size=(5, 1), padding=(2, 0))
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(channel, channel, kernel_size=(7, 1), padding=(3, 0))
        )
        self.branch4 = nn.Sequential(
            BasicConv2d(channel, channel, kernel_size=(1, 9), padding=(0, 4)),
            BasicConv2d(channel, channel, kernel_size=(9, 1), padding=(4, 0))
        )
        self.ca = ChannelAttention(channel)
        self.sa = SpatialAttention()
        self.conv_cat = BasicConv2d(channel * 4, channel, 3, padding=1)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)
        x_sa = self.sa(self.ca(x) * x)
        output = self.conv_cat(torch.cat((x_sa * x1, x_sa * x2, x_sa * x3, x_sa * x4), dim=1))

        return output

class CAEF(nn.Module):
    def __init__(self, channel):
        super(CAEF, self).__init__()
        self.Deconv = TransBasicConv2d(channel, channel, kernel_size=2, stride=2,
                                           padding=0, dilation=1, bias=False)
        self.ca1 = ChannelAttention(channel)
        self.ca2 = ChannelAttention(channel)

        self.sa1 = SpatialAttention()
        self.sa2 = SpatialAttention()
        self.avg_pool = nn.AvgPool2d((3, 3), stride=1, padding=1)
        self.max_pool = nn.MaxPool2d((3, 3), stride=1, padding=1)
        self.conv_cat = BasicConv2d(2 * channel + 1, channel, 3, padding=1)

    def forward(self, low, high):
        high = self.Deconv(high)
        high_sa = self.sa1(high) * high
        low_ca = self.ca1(low) * low
        high_to_low = self.ca2(high_sa) * low_ca
        low_to_high = self.sa2(low_ca) * high_sa
        high_low = high_to_low + low_to_high
        high_low_ap = self.avg_pool(high_low)
        high_low_mp = self.max_pool(high_low)
        edge = torch.norm(abs(high_low_ap - high_low_mp), p=2, dim=1, keepdim=True)
        x_out = self.conv_cat(torch.cat((high_to_low, low_to_high, edge), 1)) + high + low

        return x_out

class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()

        self.reduce_sal1 = Reduction(16, channel)
        self.reduce_sal2 = Reduction(24, channel)
        self.reduce_sal3 = Reduction(32, channel)
        self.reduce_sal4 = Reduction(96, channel)
        self.reduce_sal5 = Reduction(320, channel)

        self.maaf1 = MAAF(channel)
        self.maaf2 = MAAF(channel)
        self.maaf3 = MAAF(channel)
        self.maaf4 = MAAF(channel)
        self.maaf5 = MAAF(channel)

        self.caef1 = CAEF(channel)
        self.caef2 = CAEF(channel)
        self.caef3 = CAEF(channel)
        self.caef4 = CAEF(channel)
        self.caef5 = CAEF(channel)
        self.caef6 = CAEF(channel)
        self.caef7 = CAEF(channel)
        self.caef8 = CAEF(channel)
        self.caef9 = CAEF(channel)
        self.caef10 = CAEF(channel)

        self.S1 = nn.Sequential(
            BasicConv2d(channel, channel, 3, padding=1),
            nn.Conv2d(channel, 1, 1)
        )

    def forward(self, x_sal1, x_sal2, x_sal3, x_sal4, x_sal5):
        x_sal1 = self.reduce_sal1(x_sal1)
        x_sal2 = self.reduce_sal2(x_sal2)
        x_sal3 = self.reduce_sal3(x_sal3)
        x_sal4 = self.reduce_sal4(x_sal4)
        x_sal5 = self.reduce_sal5(x_sal5)

        x_sal1 = self.maaf1(x_sal1)
        x_sal2 = self.maaf2(x_sal2)
        x_sal3 = self.maaf3(x_sal3)
        x_sal4 = self.maaf4(x_sal4)
        x_sal5 = self.maaf5(x_sal5)

        x4_1 = self.caef1(x_sal4, x_sal5)
        x3_1 = self.caef2(x_sal3, x_sal4)
        x2_1 = self.caef3(x_sal2, x_sal3)
        x1_1 = self.caef4(x_sal1, x_sal2)

        x3_2 = self.caef5(x3_1, x4_1)
        x2_2 = self.caef6(x2_1, x3_1)
        x1_2 = self.caef7(x1_1, x2_1)

        x2_3 = self.caef8(x2_2, x3_2)
        x1_3 = self.caef9(x1_2, x2_2)

        x1_4 = self.caef10(x1_3, x2_3)

        sal_out = self.S1(x1_4)

        return sal_out


class SFINet(nn.Module):
    def __init__(self, pretrained=True, channel=32):

        super(SFINet, self).__init__()
        self.backbone = mobilenet_v2(pretrained)
        self.decoder = Decoder(channel)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        size = x.size()[2:]
        x_res = []
        x_sigmoid= []
        x_sal1, x_sal2, x_sal3, x_sal4, x_sal5 = self.backbone(x)
        x_out = self.decoder(x_sal1, x_sal2, x_sal3, x_sal4, x_sal5)
        sal_out = F.interpolate(x_out, size=size, mode='bilinear', align_corners=True)
        x_res.append(sal_out)
        x_sigmoid.append(self.sigmoid(sal_out))

        for cycle in range(2):
            x_sal1, x_sal2, x_sal3, x_sal4, x_sal5 = self.backbone(x, x_out)
            x_out = self.decoder(x_sal1, x_sal2, x_sal3, x_sal4, x_sal5)
            sal_out = F.interpolate(x_out, size=size, mode='bilinear', align_corners=True)
            x_res.append(sal_out)
            x_sigmoid.append(self.sigmoid(sal_out))

        return x_res, x_sigmoid