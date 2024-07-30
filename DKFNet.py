import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
__all__ = ['UNet', 'NestedUNet']


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class CoordAtt_with_ndwi(nn.Module):
    def __init__(self, inp, oup, factor=2, reduction=32):
        super(CoordAtt_with_ndwi, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        self.pool = nn.MaxPool2d(factor, factor)

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x, ndwi):
        identity = x
        ndwi = self.pool(ndwi)
        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h* ndwi

        return out

class VGGBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):    #  [b, in_channels, h, w]=>[b, out_channels, h, w]
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class VGGBlock_DCA(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, factor):    #  [b, in_channels, h, w]=>[b, out_channels, h, w]
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.dca = CoordAtt_with_ndwi(middle_channels,middle_channels,factor)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x, ndwi):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dca(out, ndwi)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class VGGBlock_CA(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):    #  [b, in_channels, h, w]=>[b, out_channels, h, w]
        super().__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, middle_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(middle_channels)
        self.ca = CoordAtt(middle_channels, middle_channels)
        self.conv2 = nn.Conv2d(middle_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.ca(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        return out

class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),                                   ##这里输出的是[B,in_channels,1,1]
            nn.Conv2d(in_channels, out_channels, 1, bias=False),       #[B,in_channels,h, w]=>[B,in_channels,1,1]=>[B,out_channels,1,1]=>[B,out_channels,h, w]
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        # print(size, "size")
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class DASPP(nn.Module):                        ##[B, in_channels, H, W] =>[B, 256, H, W]
    def __init__(self, in_channels, out_channels, factor=8):
        super(DASPP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.Nonlocal = Nonlocal(in_channels, factor=16)
        self.pool = nn.MaxPool2d(factor, factor)

        self.con1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        rate1, rate2, rate3, rate4 = (1, 3, 6, 9)    ##这里是卷积率的大小
        # self.atrous_block7 = nn.Conv2d(in_channels, out_channels, 1,  bias=False)
        # self.atrous_block2 = ASPPConv(in_channels, out_channels, rate1)
        self.atrous_block3 = ASPPConv(in_channels, out_channels, rate2)
        self.atrous_block4 = ASPPConv(in_channels, out_channels, rate3)
        self.atrous_block5 = ASPPConv(in_channels, out_channels, rate4)
        self.atrous_block6 = ASPPPooling(in_channels, out_channels)
        self.project = nn.Sequential(
            nn.Conv2d( 6* out_channels, 2 * out_channels, 1, bias=False),
            nn.BatchNorm2d(2*out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
        # self.conv = nn.Conv2d(in_channels, 1, 1, bias=False)
        # self.conv1 = nn.Conv2d(1, 1, 1, bias=False)

    def forward(self, x, ndwi):

        # atrous_block2 = self.atrous_block2(x)
        atrous_block3 = self.atrous_block3(x)
        atrous_block4 = self.atrous_block4(x)
        atrous_block5 = self.atrous_block5(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block7 = self.con1x1(x)
        ndwi = self.pool(ndwi)
        # ndwi = (np.max(ndwi) - ndwi) / (np.max(ndwi) - np.min(ndwi))
        atrous_block1 = ndwi.expand_as(atrous_block3)
        res = self.project(torch.cat((atrous_block1,  atrous_block3, atrous_block4, atrous_block5, atrous_block6, atrous_block7), dim=1))

        return res

class ASPP(nn.Module):                        ##[B, in_channels, H, W] =>[B, 256, H, W]
    def __init__(self, in_channels, out_channels, factor=8):
        super(ASPP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        # self.Nonlocal = Nonlocal(in_channels, factor=16)
        self.pool = nn.MaxPool2d(factor, factor)
        self.con1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

        rate1, rate2, rate3, rate4 = (1, 3, 6, 9)    ##这里是卷积率的大小
        # self.atrous_block1 = ASPPConv(in_channels, out_channels, rate0)
        # self.atrous_block2 = nn.Conv2d(in_channels, out_channels, 1, 1, bias=False)
        self.atrous_block3 = ASPPConv(in_channels, out_channels, rate2)
        self.atrous_block4 = ASPPConv(in_channels, out_channels, rate3)
        self.atrous_block5 = ASPPConv(in_channels, out_channels, rate4)
        self.atrous_block6 = ASPPPooling(in_channels, out_channels)
        self.project = nn.Sequential(
            nn.Conv2d( 5* out_channels, 2 * out_channels, 1, bias=False),
            nn.BatchNorm2d(2*out_channels),
            nn.ReLU(),
            nn.Dropout(0.5))
        self.conv = nn.Conv2d(in_channels, 1, 1, bias=False)
        # self.conv1 = nn.Conv2d(1, 1, 1, bias=False)

    def forward(self, x):

        atrous_block2 = self.con1x1(x)
        atrous_block3 = self.atrous_block3(x)
        atrous_block4 = self.atrous_block4(x)
        atrous_block5 = self.atrous_block5(x)
        atrous_block6 = self.atrous_block6(x)

        res = self.project(torch.cat((atrous_block2, atrous_block3, atrous_block4, atrous_block5, atrous_block6), dim=1))

        return res



class DKFNet_DASPP_without_DCA(nn.Module):
    def __init__(self, n_channels, n_classes, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.aspp0 = DASPP(512, 256, factor = 16)
        self.conv0_0 = VGGBlock(n_channels, nb_filter[0], nb_filter[0])    # [b, input_channels, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        self.conv1_0 = VGGBlock(nb_filter[0], nb_filter[1], nb_filter[1])  # [b, 32, h, w]=>[b, 64, h, w]=>[b, 64, h, w]
        self.conv2_0 = VGGBlock(nb_filter[1], nb_filter[2], nb_filter[2])  # [b, 64, h, w]=>[b, 128, h, w]=>[b, 128, h, w]
        self.conv3_0 = VGGBlock(nb_filter[2], nb_filter[3], nb_filter[3])  # [b, 128, h, w]=>[b, 256, h, w]=>[b, 256, h, w]
        self.conv4_0 = VGGBlock(nb_filter[3], nb_filter[4], nb_filter[4])  # [b, 256, h, w]=>[b, 512, h, w]=>[b, 512, h, w]

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        self.LS = nn.Sigmoid()


    def forward(self, input, ndwi):

        x0_0 = self.conv0_0(input)  # [b, input_channels, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        x1_0 = self.conv1_0(self.pool(x0_0))  # [b, 32, h, w]=>[b, 32, h/2, w/2]=>[b, 64, h/2, w/2]=>[b, 64, h/2, w/2]
        x2_0 = self.conv2_0(self.pool(x1_0))  # [b, 64, h/2, w/2]=>[b, 64, h/4, w/4]=>[b, 128, h/4, w/4]=>[b, 128, h/4, w/4]
        x3_0 = self.conv3_0(self.pool(x2_0))  # [b, 128, h/4, w/4]=>[b, 128, h/8, w/8]=>[b, 256, h/8, w/8]=>[b, 256, h/8, w/8]
        x4_0 = self.conv4_0(self.pool(x3_0))  # [b, 256, h/8, w/8]=>[b, 256, h/16, w/16]=>[b, 512, h/16, w/16]=>[b, 512, h/16, w/16]
        x4_0_aspp = self.aspp0(x4_0, ndwi)  # [b, 512, h/16, w/16]=>[b, 512, h/16, w/16]

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0_aspp)], 1))     # [b, 512, h/16, w/16]=>[b, 512, h/8, w/8] + [b, 256, h/8, w/8]=>[b, 768, h/8, w/8]=>[b, 256, h/8, w/8]=>[b, 256, h/8, w/8]                               #[b, 256, h/8, w/8]=>[b, 256, h/8, w/8]
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))     # [b, 256, h/8, w/8]=>[b, 256, h/4, w/4] + [b, 128, h/4, w/4]=>[b, 384, h/4, w/4]=>[b, 128, h/4, w/4]=>[b, 128, h/4, w/4]
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))     # [b, 128, h/4, w/4]=>[b, 128, h/2, w/2] + [b, 64, h/2, w/2]=>[b, 192, h/2, w/2]=>[b, 64, h/2, w/2]=>[b, 64, h/2, w/2]
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))     # [b, 64, h/2, w/2]=>[b, 64, h, w] + [b, 32, h, w]=>[b, 96, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        output = self.LS(self.final(x0_4))    #[b, 32, h, w]=>[b, 1, h, w]

        return output

class DKFNet_DCA_without_DASPP(nn.Module):
    def __init__(self, n_channels, n_classes, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock_DCA(n_channels, nb_filter[0], nb_filter[0], factor = 1)    # [b, input_channels, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        self.conv1_0 = VGGBlock_DCA(nb_filter[0], nb_filter[1], nb_filter[1],factor = 2)  # [b, 32, h, w]=>[b, 64, h, w]=>[b, 64, h, w]
        self.conv2_0 = VGGBlock_DCA(nb_filter[1], nb_filter[2], nb_filter[2],factor = 4)  # [b, 64, h, w]=>[b, 128, h, w]=>[b, 128, h, w]
        self.conv3_0 = VGGBlock_DCA(nb_filter[2], nb_filter[3], nb_filter[3],factor = 8)  # [b, 128, h, w]=>[b, 256, h, w]=>[b, 256, h, w]
        self.conv4_0 = VGGBlock_DCA(nb_filter[3], nb_filter[4], nb_filter[4],factor = 16)  # [b, 256, h, w]=>[b, 512, h, w]=>[b, 512, h, w]

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        self.LS = nn.Sigmoid()


    def forward(self, input, ndwi):

        x0_0 = self.conv0_0(input, ndwi)  # [b, input_channels, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        x1_0 = self.conv1_0(self.pool(x0_0), ndwi)  # [b, 32, h, w]=>[b, 32, h/2, w/2]=>[b, 64, h/2, w/2]=>[b, 64, h/2, w/2]
        x2_0 = self.conv2_0(self.pool(x1_0), ndwi)  # [b, 64, h/2, w/2]=>[b, 64, h/4, w/4]=>[b, 128, h/4, w/4]=>[b, 128, h/4, w/4]
        x3_0 = self.conv3_0(self.pool(x2_0), ndwi)  # [b, 128, h/4, w/4]=>[b, 128, h/8, w/8]=>[b, 256, h/8, w/8]=>[b, 256, h/8, w/8]
        x4_0 = self.conv4_0(self.pool(x3_0), ndwi)  # [b, 256, h/8, w/8]=>[b, 256, h/16, w/16]=>[b, 512, h/16, w/16]=>[b, 512, h/16, w/16]


        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0)], 1))     # [b, 512, h/16, w/16]=>[b, 512, h/8, w/8] + [b, 256, h/8, w/8]=>[b, 768, h/8, w/8]=>[b, 256, h/8, w/8]=>[b, 256, h/8, w/8]
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))     # [b, 256, h/8, w/8]=>[b, 256, h/4, w/4] + [b, 128, h/4, w/4]=>[b, 384, h/4, w/4]=>[b, 128, h/4, w/4]=>[b, 128, h/4, w/4]
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))     # [b, 128, h/4, w/4]=>[b, 128, h/2, w/2] + [b, 64, h/2, w/2]=>[b, 192, h/2, w/2]=>[b, 64, h/2, w/2]=>[b, 64, h/2, w/2]
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))     # [b, 64, h/2, w/2]=>[b, 64, h, w] + [b, 32, h, w]=>[b, 96, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        output = self.LS(self.final(x0_4))    #[b, 32, h, w]=>[b, 1, h, w]

        return output


class DKFNet_DASPP_DCA(nn.Module):
    def __init__(self, n_channels, n_classes, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock_DCA(n_channels, nb_filter[0], nb_filter[0], factor = 1)    # [b, input_channels, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        self.conv1_0 = VGGBlock_DCA(nb_filter[0], nb_filter[1], nb_filter[1],factor = 2)  # [b, 32, h, w]=>[b, 64, h, w]=>[b, 64, h, w]
        self.conv2_0 = VGGBlock_DCA(nb_filter[1], nb_filter[2], nb_filter[2],factor = 4)  # [b, 64, h, w]=>[b, 128, h, w]=>[b, 128, h, w]
        self.conv3_0 = VGGBlock_DCA(nb_filter[2], nb_filter[3], nb_filter[3],factor = 8)  # [b, 128, h, w]=>[b, 256, h, w]=>[b, 256, h, w]
        self.conv4_0 = VGGBlock_DCA(nb_filter[3], nb_filter[4], nb_filter[4],factor = 16)  # [b, 256, h, w]=>[b, 512, h, w]=>[b, 512, h, w]

        self.aspp4 = DASPP(512, 256, factor=16)

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        self.LS = nn.Sigmoid()


    def forward(self, input, ndwi):

        x0_0 = self.conv0_0(input, ndwi)  # [b, input_channels, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        x1_0 = self.conv1_0(self.pool(x0_0), ndwi)  # [b, 32, h, w]=>[b, 32, h/2, w/2]=>[b, 64, h/2, w/2]=>[b, 64, h/2, w/2]
        x2_0 = self.conv2_0(self.pool(x1_0), ndwi)  # [b, 64, h/2, w/2]=>[b, 64, h/4, w/4]=>[b, 128, h/4, w/4]=>[b, 128, h/4, w/4]
        x3_0 = self.conv3_0(self.pool(x2_0), ndwi)  # [b, 128, h/4, w/4]=>[b, 128, h/8, w/8]=>[b, 256, h/8, w/8]=>[b, 256, h/8, w/8]
        x4_0 = self.conv4_0(self.pool(x3_0), ndwi)  # [b, 256, h/8, w/8]=>[b, 256, h/16, w/16]=>[b, 512, h/16, w/16]=>[b, 512, h/16, w/16]
        x4_0_aspp = self.aspp4(x4_0,ndwi)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0_aspp)], 1))     # [b, 512, h/16, w/16]=>[b, 512, h/8, w/8] + [b, 256, h/8, w/8]=>[b, 768, h/8, w/8]=>[b, 256, h/8, w/8]=>[b, 256, h/8, w/8]
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))     # [b, 256, h/8, w/8]=>[b, 256, h/4, w/4] + [b, 128, h/4, w/4]=>[b, 384, h/4, w/4]=>[b, 128, h/4, w/4]=>[b, 128, h/4, w/4]
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))     # [b, 128, h/4, w/4]=>[b, 128, h/2, w/2] + [b, 64, h/2, w/2]=>[b, 192, h/2, w/2]=>[b, 64, h/2, w/2]=>[b, 64, h/2, w/2]
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))     # [b, 64, h/2, w/2]=>[b, 64, h, w] + [b, 32, h, w]=>[b, 96, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        output = self.LS(self.final(x0_4))    #[b, 32, h, w]=>[b, 1, h, w]

        return output

class DKFNet_DCA_with_ASPP(nn.Module):
    def __init__(self, n_channels, n_classes, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock_DCA(n_channels, nb_filter[0], nb_filter[0], factor = 1)    # [b, input_channels, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        self.conv1_0 = VGGBlock_DCA(nb_filter[0], nb_filter[1], nb_filter[1],factor = 2)  # [b, 32, h, w]=>[b, 64, h, w]=>[b, 64, h, w]
        self.conv2_0 = VGGBlock_DCA(nb_filter[1], nb_filter[2], nb_filter[2],factor = 4)  # [b, 64, h, w]=>[b, 128, h, w]=>[b, 128, h, w]
        self.conv3_0 = VGGBlock_DCA(nb_filter[2], nb_filter[3], nb_filter[3],factor = 8)  # [b, 128, h, w]=>[b, 256, h, w]=>[b, 256, h, w]
        self.conv4_0 = VGGBlock_DCA(nb_filter[3], nb_filter[4], nb_filter[4],factor = 16)  # [b, 256, h, w]=>[b, 512, h, w]=>[b, 512, h, w]

        self.aspp4 = ASPP(512, 256, factor=16)

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        self.LS = nn.Sigmoid()


    def forward(self, input, ndwi):

        x0_0 = self.conv0_0(input, ndwi)  # [b, input_channels, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        x1_0 = self.conv1_0(self.pool(x0_0), ndwi)  # [b, 32, h, w]=>[b, 32, h/2, w/2]=>[b, 64, h/2, w/2]=>[b, 64, h/2, w/2]
        x2_0 = self.conv2_0(self.pool(x1_0), ndwi)  # [b, 64, h/2, w/2]=>[b, 64, h/4, w/4]=>[b, 128, h/4, w/4]=>[b, 128, h/4, w/4]
        x3_0 = self.conv3_0(self.pool(x2_0), ndwi)  # [b, 128, h/4, w/4]=>[b, 128, h/8, w/8]=>[b, 256, h/8, w/8]=>[b, 256, h/8, w/8]
        x4_0 = self.conv4_0(self.pool(x3_0), ndwi)  # [b, 256, h/8, w/8]=>[b, 256, h/16, w/16]=>[b, 512, h/16, w/16]=>[b, 512, h/16, w/16]
        x4_0_aspp = self.aspp4(x4_0)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0_aspp)], 1))     # [b, 512, h/16, w/16]=>[b, 512, h/8, w/8] + [b, 256, h/8, w/8]=>[b, 768, h/8, w/8]=>[b, 256, h/8, w/8]=>[b, 256, h/8, w/8]
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))     # [b, 256, h/8, w/8]=>[b, 256, h/4, w/4] + [b, 128, h/4, w/4]=>[b, 384, h/4, w/4]=>[b, 128, h/4, w/4]=>[b, 128, h/4, w/4]
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))     # [b, 128, h/4, w/4]=>[b, 128, h/2, w/2] + [b, 64, h/2, w/2]=>[b, 192, h/2, w/2]=>[b, 64, h/2, w/2]=>[b, 64, h/2, w/2]
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))     # [b, 64, h/2, w/2]=>[b, 64, h, w] + [b, 32, h, w]=>[b, 96, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        output = self.LS(self.final(x0_4))    #[b, 32, h, w]=>[b, 1, h, w]

        return output

class DKFNet_DASPP_with_CA(nn.Module):
    def __init__(self, n_channels, n_classes, **kwargs):
        super().__init__()

        nb_filter = [32, 64, 128, 256, 512]
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = VGGBlock_CA(n_channels, nb_filter[0], nb_filter[0])    # [b, input_channels, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        self.conv1_0 = VGGBlock_CA(nb_filter[0], nb_filter[1], nb_filter[1])  # [b, 32, h, w]=>[b, 64, h, w]=>[b, 64, h, w]
        self.conv2_0 = VGGBlock_CA(nb_filter[1], nb_filter[2], nb_filter[2])  # [b, 64, h, w]=>[b, 128, h, w]=>[b, 128, h, w]
        self.conv3_0 = VGGBlock_CA(nb_filter[2], nb_filter[3], nb_filter[3])  # [b, 128, h, w]=>[b, 256, h, w]=>[b, 256, h, w]
        self.conv4_0 = VGGBlock_CA(nb_filter[3], nb_filter[4], nb_filter[4])  # [b, 256, h, w]=>[b, 512, h, w]=>[b, 512, h, w]

        self.aspp4 = DASPP(512, 256, factor=16)

        self.conv3_1 = VGGBlock(nb_filter[3]+nb_filter[4], nb_filter[3], nb_filter[3])
        self.conv2_2 = VGGBlock(nb_filter[2]+nb_filter[3], nb_filter[2], nb_filter[2])
        self.conv1_3 = VGGBlock(nb_filter[1]+nb_filter[2], nb_filter[1], nb_filter[1])
        self.conv0_4 = VGGBlock(nb_filter[0]+nb_filter[1], nb_filter[0], nb_filter[0])

        self.final = nn.Conv2d(nb_filter[0], n_classes, kernel_size=1)
        self.LS = nn.Sigmoid()


    def forward(self, input, ndwi):

        x0_0 = self.conv0_0(input)  # [b, input_channels, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        x1_0 = self.conv1_0(self.pool(x0_0))  # [b, 32, h, w]=>[b, 32, h/2, w/2]=>[b, 64, h/2, w/2]=>[b, 64, h/2, w/2]
        x2_0 = self.conv2_0(self.pool(x1_0))  # [b, 64, h/2, w/2]=>[b, 64, h/4, w/4]=>[b, 128, h/4, w/4]=>[b, 128, h/4, w/4]
        x3_0 = self.conv3_0(self.pool(x2_0))  # [b, 128, h/4, w/4]=>[b, 128, h/8, w/8]=>[b, 256, h/8, w/8]=>[b, 256, h/8, w/8]
        x4_0 = self.conv4_0(self.pool(x3_0))  # [b, 256, h/8, w/8]=>[b, 256, h/16, w/16]=>[b, 512, h/16, w/16]=>[b, 512, h/16, w/16]
        x4_0_aspp = self.aspp4(x4_0,ndwi)

        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0_aspp)], 1))     # [b, 512, h/16, w/16]=>[b, 512, h/8, w/8] + [b, 256, h/8, w/8]=>[b, 768, h/8, w/8]=>[b, 256, h/8, w/8]=>[b, 256, h/8, w/8]
        x2_2 = self.conv2_2(torch.cat([x2_0, self.up(x3_1)], 1))     # [b, 256, h/8, w/8]=>[b, 256, h/4, w/4] + [b, 128, h/4, w/4]=>[b, 384, h/4, w/4]=>[b, 128, h/4, w/4]=>[b, 128, h/4, w/4]
        x1_3 = self.conv1_3(torch.cat([x1_0, self.up(x2_2)], 1))     # [b, 128, h/4, w/4]=>[b, 128, h/2, w/2] + [b, 64, h/2, w/2]=>[b, 192, h/2, w/2]=>[b, 64, h/2, w/2]=>[b, 64, h/2, w/2]
        x0_4 = self.conv0_4(torch.cat([x0_0, self.up(x1_3)], 1))     # [b, 64, h/2, w/2]=>[b, 64, h, w] + [b, 32, h, w]=>[b, 96, h, w]=>[b, 32, h, w]=>[b, 32, h, w]
        output = self.LS(self.final(x0_4))    #[b, 32, h, w]=>[b, 1, h, w]

        return output
