import torch
from torch import nn
import torch.nn.functional as F
import math
from carsot.utils.xcorr import xcorr_depthwise

class MaskHead(torch.nn.Module):
    def __init__(self, cfg, in_channels):
        """
        Arguments:
            in_channels (int): number of channels of the input feature  256
        """
        super(MaskHead, self).__init__()
        # TODO: Implement the sigmoid version first.
        mask_out = cfg.MASK.KWARGS.out_channels  # 63 * 63
        # mask_out = 3969  # 63 * 63

        mask_tower = []
        for i in range(cfg.TRAIN.NUM_CONVS):  # 4
            mask_tower.append(
                nn.Conv2d(
                    in_channels,  # 256
                    in_channels,  # 256
                    kernel_size=3,
                    stride=1,
                    padding=1
                )
            )
            mask_tower.append(nn.GroupNorm(32, in_channels))
            mask_tower.append(nn.ReLU())
        self.add_module('mask_tower', nn.Sequential(*mask_tower))
        self.mask_pred = nn.Conv2d(
            in_channels, mask_out, kernel_size=1, stride=1)

        # initialization
        for modules in [self.mask_tower, self.mask_pred]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        # self.head=DepthwiseXCorr(in_channels, in_channels, mask_out)

    def forward(self, x):
        mask_tower = self.mask_tower(x)
        mask = self.mask_pred(mask_tower)
        return mask

    # def forward(self, z, x):
    #     mask = self.head(z, x)
    #     return mask


class Refine(nn.Module):
    def __init__(self):
        """
        kernel_size=3, padding=1,尺寸不变
        只改变通道数
        """
        super(Refine, self).__init__()
        self.v0 = nn.Sequential(
                nn.Conv2d(64, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 4, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.v1 = nn.Sequential(
                nn.Conv2d(256, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.v2 = nn.Sequential(
                nn.Conv2d(512, 128, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 32, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.h2 = nn.Sequential(
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.h1 = nn.Sequential(
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(inplace=True),
            )
        self.h0 = nn.Sequential(
                nn.Conv2d(4, 4, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(4, 4, 3, padding=1),
                nn.ReLU(inplace=True),
            )

        self.deconv = nn.ConvTranspose2d(256, 32, 15, 15)
        self.post0 = nn.Conv2d(32, 16, 3, padding=1)  # 通道变换
        self.post1 = nn.Conv2d(16, 4, 3, padding=1)
        self.post2 = nn.Conv2d(4, 1, 3, padding=1)

    def forward(self, f, corr_feature, pos=None, debug=False):
        """
        pad(in, pad_tuple(left,right,top,bottom))
        :param f: xf list of search out of backbone
         layer1:[1, 64, 125, 125], layer2:[1, 256, 63, 63]  layer3:[1, 512, 31, 31]
        :param corr_feature: corr_feature of mask branch (bs, 256, 25,25)
        :param pos: position of best_score  (y,x) if tracking
        :return:
        """
        if debug:
            print('********refine********')
            print('xf.shape=', len(f))
            print('xf[0].shape={} | xf[1].shape={} | xf[2].shape={}'.format(
                f[0].shape, f[1].shape, f[2].shape))
            print('corr_feature.shape=', corr_feature.shape)
        if pos is not None:
            p0 = F.pad(f[0], (16, 16, 16, 16))[:, :, 4*pos[0]:4*pos[0]+61, 4*pos[1]:4*pos[1]+61]  # layer1 [1, 64, 61, 61]
            p1 = F.pad(f[1], (8, 8, 8, 8))[:, :, 2*pos[0]:2*pos[0]+31, 2*pos[1]:2*pos[1]+31]  # layer2  [1, 256, 31, 31]
            p2 = F.pad(f[2], (4, 4, 4, 4))[:, :, pos[0]:pos[0]+15, pos[1]:pos[1]+15]   # layer3  [1, 512, 15, 15]
            if debug:
                print('p0.shape={} | p1.shape={} | p2.shape={}'.format(
                    p0.shape, p1.shape, p2.shape))
        else:
            # [B*17*17=289, C, kH, kW]
            p0 = F.unfold(f[0], (61, 61), padding=0, stride=4)\
                .permute(0, 2, 1).contiguous().view(-1, 64, 61, 61)   # [bs*289, 64, 61, 61]
            p1 = F.unfold(f[1], (31, 31), padding=0, stride=2)\
                .permute(0, 2, 1).contiguous().view(-1, 256, 31, 31)  # [bs*289, 256, 31, 31]
            p2 = F.unfold(f[2], (15, 15), padding=0, stride=1)\
                .permute(0, 2, 1).contiguous().view(-1, 512, 15, 15)  # [bs*289, 512, 15, 15]
        if debug:
            print('p0.shape={} | p1.shape={} | p2.shape={}'.format(
                p0.shape, p1.shape, p2.shape))

        if pos is not None:
            # (1, 256, 25,25)
            p3 = corr_feature[:, :, pos[0], pos[1]].view(-1, 256, 1, 1)   # 获取最佳位置的相似度向量
        else:
            # (bs, 256, 25,25)->(bs,25,25,256)->(bs*625, 256, 1, 1)
            p3 = corr_feature.permute(0, 2, 3, 1).contiguous().view(-1, 256, 1, 1)
        assert p3.shape[0] == p2.shape[0]

        # [1, 32, 15, 15] if tracking
        out = self.deconv(p3)
        if debug: print('deconv out.shape=', out.shape)

        out = self.post0(F.upsample(self.h2(out) + self.v2(p2), size=(31, 31)))   # [1, 16, 31, 31]
        if debug: print('layer3 out.shape=', out.shape)

        out = self.post1(F.upsample(self.h1(out) + self.v1(p1), size=(61, 61)))   # [1, 4, 61, 61]
        if debug: print('layer2 out.shape=', out.shape)

        out = self.post2(F.upsample(self.h0(out) + self.v0(p0), size=(127, 127)))  # [1, 1, 127, 127]
        if debug: print('layer1 out.shape=', out.shape)
        out = out.view(-1, 127*127)
        return out


class DepthwiseXCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthwiseXCorr, self).__init__()
        # adj_1
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        # adj_2
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        # Box_head or Cls_head 256 -> 2k or 4k = (10, 20)
        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),  # fuse different channel outputs
            nn.BatchNorm2d(hidden),

            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)  # RPN自动产生proposal的[po,ne, | dx,dy,dw,dh]
        )

    def forward(self, kernel, search):
        kernel = self.conv_kernel(kernel)  # [-1, 256, 5, 5]   # adj_1
        search = self.conv_search(search)  # [-1, 256, 29, 29] # adj_2
        feature = xcorr_depthwise(search, kernel)  # DW_Corr
        out = self.head(feature)  # Box_head or Cls_head
        return out

from carsot.core.config import cfg
from torchsummary import summary

if __name__ == '__main__':
    net = MaskHead(cfg, 256).to('cuda')
    print(net)
    summary(net, (256, 25, 25))