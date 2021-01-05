import torch.nn as nn
from torchsummary import summary
import torch
from carsot.core.config import cfg

class AdjustLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustLayer, self).__init__()
        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        x = self.downsample(x)
        if cfg.REFINE.REFINE:
            # SiamMask
            if x.size(3) < 20:    # 15 < 20 ->(7,7)
                l = 4
                r = -4
                x = x[:, :, l:r, l:r]   # len([4:-4]) 不一定等于 7
        else:
            if x.size(3) < 20:  # height < 20  template 15 < 20
                l = 4
                r = l + 7
                x = x[:, :, l:r, l:r]  # x[:, :, 4:11, 4:11] 7*7
        return x


class AdjustAllLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdjustAllLayer, self).__init__()
        self.num = len(out_channels)  # len([256, 256, 256]) = 3
        if self.num == 1:
            self.downsample = AdjustLayer(in_channels[0], out_channels[0])
        else:
            for i in range(self.num):
                self.add_module('downsample'+str(i+2),   # layer2, 3, 4
                                AdjustLayer(in_channels[i], out_channels[i]))

    def forward(self, features):
        if self.num == 1:
            return self.downsample(features)
        else:
            out = []
            for i in range(self.num):
                adj_layer = getattr(self, 'downsample'+str(i+2))
                # contiguous:transpose,permute操作后保证底层数组行优先一维展开，使view()正常执行
                out.append(adj_layer(features[i]).contiguous())
            return out


if __name__ == '__main__':
    net = AdjustLayer(2048, 256).to('cuda')
    print(net)
    x = (1, 2048, 15, 15)
    summary(net, x[1:]) # 只能打印nn.module里输出size, 不能打印自定义函数的输出
    sample = torch.rand(x).to('cuda')
    out = net(sample)  # torch.Size([1, 256, 7, 7]
    print('***********out=', out.shape)

    in_channels = [512, 1024, 2048]
    out_channels = [256, 256, 256]
    net2 = AdjustAllLayer(in_channels, out_channels)
    print(net2)