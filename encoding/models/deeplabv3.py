from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsnooper
from .fcn import FCNHead
from .base import BaseNet
from encoding.nn import SyncBatchNorm
__all__ = ['DeepLabV3', 'get_deeplab']

class DeepLabV3(BaseNet):

    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DeepLabV3, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = DeepLabV3Head(2048, nclass, norm_layer, self._up_kwargs)


        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, SyncBatchNorm):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        _, _, h, w = x.size()

        _, _, c3, c4 = self.base_forward(x)


        outputs = []
        x = self.head(c4)


        x = F.interpolate(x, (h,w), **self._up_kwargs)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h,w), **self._up_kwargs)
            outputs.append(auxout)

        return tuple(outputs)


class DeepLabV3Head(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs, atrous_rates=(12, 24, 36)):
        super(DeepLabV3Head, self).__init__()
        inter_channels = in_channels // 8
        # self.aspp = ASPP_Module(in_channels, atrous_rates, norm_layer, up_kwargs)
        self.aspp = MSPP_Module(in_channels, atrous_rates, norm_layer, up_kwargs)
        self.block = nn.Sequential(
            # nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        x = self.aspp(x)
        x = self.block(x)
        return x



def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block

class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(AsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return F.interpolate(pool, (h,w), **self._up_kwargs)
class MsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(MsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels)
                                 )

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return pool
        # return F.interpolate(pool, (h,w), **self._up_kwargs)
class ASPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
        super(ASPP_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(5*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.5, False))

    def forward(self, x):
        
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)

        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)

        return self.project(y)

class MSPP_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
        super(MSPP_Module, self).__init__()
        out_channels = in_channels // 8
        self.channels = in_channels
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))


        # self.conv1x3 =nn.Sequential(nn.Conv2d(in_channels, out_channels, (1,3), padding=(0,1),
        #                       bias=False),
        #                         norm_layer(out_channels)
        #                         # ,nn.ReLU(True)
        #                         )

        # self.conv3x1 =nn.Sequential(nn.Conv2d(in_channels, out_channels, (3,1), padding=(1,0),
        #                       bias=False),
        #                         norm_layer(out_channels)
        #                         # ,nn.ReLU(True)
        #                         )
        self.conv3x3 =nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1,
                               bias=False),
                                norm_layer(out_channels)
                                ,nn.ReLU(True)
                                )

        self.conv3x5 =nn.Sequential(nn.Conv2d(in_channels, out_channels, (3,5), padding=(1,2),
                               bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))


        self.conv5x3 =nn.Sequential(nn.Conv2d(in_channels, out_channels, (5,3), padding=(2,1),
                               bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv5x5 =nn.Sequential(
            					nn.Conv2d(in_channels, out_channels, (5,5), padding=(2,2),
                               bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        # self.conv1x7 =nn.Sequential(nn.Conv2d(in_channels, out_channels, (1,7), padding=(0,3),
        #                        bias=False),
        #                         norm_layer(out_channels),
        #                         nn.ReLU(True))
        # self.conv7x1 =nn.Sequential(nn.Conv2d(in_channels, out_channels, (7,1), padding=(3,0),
        #                        bias=False),
        #                         norm_layer(out_channels),
        #                         nn.ReLU(True))
        # self.conv7x7 =nn.Sequential(nn.Conv2d(in_channels, out_channels, (7,7), padding=(3,3),
        #                        bias=False),
        #                         norm_layer(out_channels),
        #                         nn.ReLU(True))
        # self.assp_pool = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)
        self.mssp_pool = MsppPooling(in_channels, out_channels, norm_layer, up_kwargs)
        # self.project1 = nn.Sequential(
        #     nn.Conv2d(3*out_channels, out_channels, 1, bias=False),
        #     norm_layer(out_channels),
        #     nn.ReLU(True),
        #     nn.Dropout2d(0.5, False))
        # self.project2 = nn.Sequential(
        #     nn.Conv2d(3*out_channels, out_channels, 1, bias=False),
        #     norm_layer(out_channels),
        #     nn.ReLU(True),
        #     nn.Dropout2d(0.5, False))
        # self.project3 = nn.Sequential(
        #     nn.Conv2d(2*out_channels, out_channels, 1, bias=False),
        #     norm_layer(out_channels),
        #     nn.ReLU(True),
        #     nn.Dropout2d(0.5, False))

        self.project = nn.Sequential(
            nn.Conv2d(6*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.5, False))
        # self.relu = nn.ReLU(inplace=True)
        # self.ACB_relu = nn.ReLU(inplace=True)
    #@torchsnooper.snoop()
    def forward(self, x):

        conv1x1 = self.conv1x1(x)
        conv3x3 = self.conv3x3(x)
        conv5x5 = self.conv5x5(x)
        # conv7x7 = self.conv7x7(x)

        # conv1x3 = self.conv1x3(x)
        # conv3x1 = self.conv3x1(x)
        # ACBlock = self.ACB_relu(conv3x1+conv1x3+conv3x3)

        conv3x5 = self.conv3x5(x)
        conv5x3 = self.conv5x3(x)

        # conv1x7 = self.conv1x7(x)
        # conv7x1 = self.conv7x1(x)
        gap = self.mssp_pool(x)
        # gap = self.assp_pool(x)
        c_gap = torch.mean(x,dim=1,keepdim=True)
        
        # x_gpb = nn.AvgPool2d(x.shape[2:])(x).view(x.shape[0], 256, 1, 1)
        # print(gap.shape,x_gpb.shape)
        # g = gap+channel_gap

        # print(g.shape)

        # feat6 = self.b6(x)
        # print(feat0.size(),feat1.size(),feat2.size(),feat3.size(),feat4.size())
        # exit(0)

        # feat0 = conv1x1
        # feat1 = self.project1(torch.cat((conv1x3,conv3x1,conv3x3), 1))
        # feat2 = self.project2(torch.cat((conv3x5,conv5x3,conv5x5), 1))
        # feat3 = self.project3(torch.cat((conv1x7,conv7x1), 1))

        # y = gap*self.project(torch.cat((conv1x1,conv3x3,conv5x5,conv7x7), 1))
        # y = self.relu(y)

        # res = y*assp_pool
        # y = self.project(torch.cat((conv3x3,conv3x5,conv5x3,conv5x5), 1))
        # gap = self.aspp(x)
        y = self.project(torch.cat((conv1x1,conv3x3,conv3x5,conv5x3,conv5x5,gap), 1))

        # y = torch.cat((conv3x3,conv3x5,conv5x3,conv5x5), 1)

        
        return (c_gap+conv1x1)*y+gap
        

        # return conv1x1*y+gap
        # return y

def get_deeplab(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = DeepLabV3(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model
