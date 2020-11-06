import math

import torch.nn as nn
import torch
from torchsummary import summary
"""
 (224; 256; 384; 480; 640g)
原始的resnet支持输入尺寸(224, 224), 不支持(255, 255),所以得该
"""
__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50']

# shift invariant
# def conv3x3(in_planes, out_planes, stride=1, groups=1):
#     """3x3 convolution with padding"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
#                  padding=1, groups=groups, bias=False)
#
# def conv1x1(in_planes, out_planes, stride=1):
#     """1x1 convolution"""
#     return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)
#
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, filter_size=1):
#         super(BasicBlock, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         if groups != 1:
#             raise ValueError('BasicBlock only supports groups=1')
#         # Both self.conv1 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv3x3(inplanes, planes)
#         self.bn1 = norm_layer(planes)
#         self.relu = nn.ReLU(inplace=True)
#         if(stride==1):
#             self.conv2 = conv3x3(planes,planes)
#         else:
#             self.conv2 = nn.Sequential(Downsample(filt_size=filter_size, stride=stride, channels=planes),
#                 conv3x3(planes, planes),)
#         self.bn2 = norm_layer(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class Bottleneck(nn.Module):
#     expansion = 4
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, norm_layer=None, filter_size=1):
#         super(Bottleneck, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         # Both self.conv2 and self.downsample layers downsample the input when stride != 1
#         self.conv1 = conv1x1(inplanes, planes)
#         self.bn1 = norm_layer(planes)
#         self.conv2 = conv3x3(planes, planes, groups) # stride moved
#         self.bn2 = norm_layer(planes)
#         if(stride==1):
#             self.conv3 = conv1x1(planes, planes * self.expansion)
#         else:
#             self.conv3 = nn.Sequential(Downsample(filt_size=filter_size, stride=stride, channels=planes),
#                 conv1x1(planes, planes * self.expansion))
#         self.bn3 = norm_layer(planes * self.expansion)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out
#
#
# class ResNet(nn.Module):
#
#     def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
#                  groups=1, width_per_group=64, norm_layer=None, filter_size=1, pool_only=True):
#         super(ResNet, self).__init__()
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         planes = [int(width_per_group * groups * 2 ** i) for i in range(4)]
#         self.inplanes = planes[0]
#
#         if(pool_only):
#             self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=2, padding=3, bias=False)
#         else:
#             self.conv1 = nn.Conv2d(3, planes[0], kernel_size=7, stride=1, padding=3, bias=False)
#         self.bn1 = norm_layer(planes[0])
#         self.relu = nn.ReLU(inplace=True)
#
#         if(pool_only):
#             self.maxpool = nn.Sequential(*[nn.MaxPool2d(kernel_size=2, stride=1),
#                 Downsample(filt_size=filter_size, stride=2, channels=planes[0])])
#         else:
#             self.maxpool = nn.Sequential(*[Downsample(filt_size=filter_size, stride=2, channels=planes[0]),
#                 nn.MaxPool2d(kernel_size=2, stride=1),
#                 Downsample(filt_size=filter_size, stride=2, channels=planes[0])])
#
#         self.layer1 = self._make_layer(block, planes[0], layers[0], groups=groups, norm_layer=norm_layer)
#         self.layer2 = self._make_layer(block, planes[1], layers[1], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
#         self.layer3 = self._make_layer(block, planes[2], layers[2], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
#         self.layer4 = self._make_layer(block, planes[3], layers[3], stride=2, groups=groups, norm_layer=norm_layer, filter_size=filter_size)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(planes[3] * block.expansion, num_classes)
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 if(m.in_channels!=m.out_channels or m.out_channels!=m.groups or m.bias is not None):
#                     # don't want to reinitialize downsample layers, code assuming normal conv layers will not have these characteristics
#                     nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 else:
#                     print('Not initializing')
#             elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
#                 nn.init.constant_(m.weight, 1)
#                 nn.init.constant_(m.bias, 0)
#
#         # Zero-initialize the last BN in each residual branch,
#         # so that the residual branch starts with zeros, and each residual block behaves like an identity.
#         # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
#         if zero_init_residual:
#             for m in self.modules():
#                 if isinstance(m, Bottleneck):
#                     nn.init.constant_(m.bn3.weight, 0)
#                 elif isinstance(m, BasicBlock):
#                     nn.init.constant_(m.bn2.weight, 0)
#
#     def _make_layer(self, block, planes, blocks, stride=1, groups=1, norm_layer=None, filter_size=1):
#         if norm_layer is None:
#             norm_layer = nn.BatchNorm2d
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             # downsample = nn.Sequential(
#             #     conv1x1(self.inplanes, planes * block.expansion, stride, filter_size=filter_size),
#             #     norm_layer(planes * block.expansion),
#             # )
#
#             downsample = [Downsample(filt_size=filter_size, stride=stride, channels=self.inplanes),] if(stride !=1) else []
#             downsample += [conv1x1(self.inplanes, planes * block.expansion, 1),
#                 norm_layer(planes * block.expansion)]
#             # print(downsample)
#             downsample = nn.Sequential(*downsample)
#
#         layers = []
#         layers.append(block(self.inplanes, planes, stride, downsample, groups, norm_layer, filter_size=filter_size))
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(block(self.inplanes, planes, groups=groups, norm_layer=norm_layer, filter_size=filter_size))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.maxpool(x)
#
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#
#         x = self.avgpool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#
#         return x


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    "3x3 convolution with padding padding=dilation"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False, dilation=dilation)


class BasicBlock(nn.Module):
    """
    BasicBlock的stride尺寸缩放在第一个卷积层
    两个卷积层都是膨胀卷积
    第一个block的conv1和downsample: dilation==pading== 原始参数//2
    非第一个block的conv1: dilation==pading== 原始参数

    conv2: dilation==pading== 原始参数
    """
    expansion = 1

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(BasicBlock, self).__init__()
        padding = 2 - stride

        if dilation > 1:
            padding = dilation

        dd = dilation
        pad = padding
        if downsample is not None and dilation > 1:
            dd = dilation // 2
            pad = dd

        # 当dilation大于１且有downsample，　第一个block的pad==dd==dilation//2
        # layer3第一个block: cnnSize(stride=1, padding=1, dilation=1) == downsample
        # layer4第一个block: cnnSize(stride=1, padding=2, dilation=2) == downsample

        # 当dilation大于１且无downsample，　非第一个block的padding==dilation
        # layer3非第一个block: cnnSize(stride=1, padding=2, dilation=2)
        # layer4非第一个block: cnnSize(stride=1, padding=4, dilation=4)
        self.conv1 = nn.Conv2d(inplanes, planes,
                               stride=stride, dilation=dd, bias=False,
                               kernel_size=3, padding=pad)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        # conv2在每层的dilation都是原始的参数
        self.conv2 = conv3x3(planes, planes, dilation=dilation)  # 输入的dilation
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)   # dilationCNN
        out = self.bn1(out)
        out = self.relu(out)  # dilationCNN

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """
    Bottleneck的stride尺寸缩放在第二个卷积层conv2
    只有conv2是膨胀卷积
    第一个block的conv1和downsample: dilation==pading== 原始参数//2
    非第一个block的conv1: dilation==pading== 原始参数
    """
    expansion = 4

    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        padding = 2 - stride
        if downsample is not None and dilation > 1:
            # 只有每层的第一个block带downsample
            # layer3: dilation=2, 故第一个block dilation=1=padding
            # layer4: dilation=4, 故第一个block dilation=2=padding
            dilation = dilation // 2
            padding = dilation

        assert stride == 1 or dilation == 1, \
            "stride and dilation must have one equals to zero at least"

        if dilation > 1:
            padding = dilation

        # layer2所有block: cnnSize(stride=2, padding=0, dilation=1) == downsample

        # 当dilation大于１且有downsample，　第一个block的padding==dilation=dilation //2
        # layer3第一个block: cnnSize(stride=1, padding=1, dilation=1) == downsample
        # layer4第一个block: cnnSize(stride=1, padding=2, dilation=2) == downsample

        # 当dilation大于１且无downsample，　非第一个block的padding==dilation
        # layer3非第一个block: cnnSize(stride=1, padding=2, dilation=2)
        # layer4非第一个block: cnnSize(stride=1, padding=4, dilation=4)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,  # img.shape/stride
                               padding=padding, bias=False, dilation=dilation)  # dilation

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)  # *4
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)  # dilationCNN
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        out = self.relu(out)

        return out


class ResNet(nn.Module):
    """
    layer1 stride=1, 但是 maxpool和conv1使输入x的shape共/4
    layer2 stride=2,图像尺寸减半
    layer3and4 stride=1,尺寸不变,

    dilationCNN
    只在layer3and4进行膨胀卷积
    layer3: dilation=2==padding,  real receptive field of 5x5
    layer4: dilation=4==padding,  real receptive field of 9x9

    当k=3, stride=1:out = floor[(x + 2p - 2d - 1) + 1]
    只要 padding==dilation 则　out = x, 输出尺寸不变
    所以每层的dilationConv2d的输出尺寸不变

    out.shape = x // 8 , 255 // 8 = 31
    """
    def __init__(self, block, layers, used_layers):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=0,  # 3 # /2
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # /2
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # /2

        self.feature_size = 128 * block.expansion
        self.used_layers = used_layers
        layer3 = True if 3 in used_layers else False
        layer4 = True if 4 in used_layers else False

        if layer3:
            self.layer3 = self._make_layer(block, 256, layers[2],
                                           stride=1, dilation=2)  # 5x5  # /1
            self.feature_size = (256 + 128) * block.expansion  # 1536
        else:
            self.layer3 = lambda x: x  # identity

        if layer4:
            self.layer4 = self._make_layer(block, 512, layers[3],
                                           stride=1, dilation=4)  # 9x9 # /1
            self.feature_size = 512 * block.expansion  # 2048
        else:
            self.layer4 = lambda x: x  # identity

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        """
        downsample(): 就是通道转换 + 尺寸缩放(stride, padding ,dilation控制)。
        更准确地说, 每层的第一个Bottleneck的conv2和downsample()有相同的卷积参数
        cnnSize(kernel_size, stride, padding, dilation)

        使转换后的输入能与block的最后一层输出有相同的通道数和尺寸，以便求和.
        且执行于每层的第一个block的最后一层.
        Bottleneck的stride尺寸缩放在第二个卷积层
        e.g.
        输入x = [-1, 256, 64, 64]
        block的最后一个卷积层输出是y = [-1, 512, 32, 32]
        block的out = downsample(x) + y with downsample(x) = [-1, 512, 32, 32]

        :param block:  block 类型
        :param planes: block的前两个卷积层的输出尺寸,但不是最后的block输出channel
        :param blocks: block个数
        :param stride: 层与层之间的过渡，即每层的第一个block的stride,一般为２降采样
        控制每层图片的尺寸.
        :param dilation:
        :return:Sequential with channel = planes * block.expansion,
        """
        downsample = None
        dd = dilation
        # bottleneck: 64 != 64 * 4, 第一层有通道变换 downsample
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1 and dilation == 1:
                # 执行于第一层的block的输出
                # 无dilation的通道变换
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
            else:
                if dilation > 1:
                    # 确定padding
                    # 发生于第三四层的第一个block的输出
                    # layer3 dd=2//2=1 cnnSize(stride=1, padding=1, dilation=1)
                    # layer4 dd=4//2=2 cnnSize(stride=1, padding=2, dilation=2)
                    dd = dilation // 2
                    padding = dd

                else:
                    # 该层内的stride > 1,
                    # 发生在第二层的第一个block的输出,只有第二层stride=2>1
                    # layer2第一个block: cnnSize(stride=2, padding=0, dilation=1)
                    dd = 1
                    padding = 0

                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion,
                              kernel_size=3, stride=stride, bias=False,  # kernel=3
                              padding=padding, dilation=dd),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        layers.append(block(self.inplanes, planes, stride,
                            downsample, dilation=dilation))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)    # /2
        x = self.bn1(x)
        x_ = self.relu(x)
        x = self.maxpool(x_)  # [-1, 64, 63, 63]  # /2

        p1 = self.layer1(x)   # [-1, 64, 63, 63]
        p2 = self.layer2(p1)  # [-1, 128, 63, 63] # /2
        p3 = self.layer3(p2)  # [-1, 1024, 31, 31]
        p4 = self.layer4(p3)  # [-1, 2048, 31, 31]
        out = [x_, p1, p2, p3, p4]
        # 从out中选择一些层的输出，作为输出列表
        out = [out[i] for i in self.used_layers]
        if len(out) == 1:
            return out[0]
        else:
            return out
        # return p4


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.

    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.

    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def resnet50(**kwargs):
    """Constructs a ResNet-50 model.

    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


if __name__ == '__main__':
    net = resnet50(used_layers=[2, 3, 4]).to('cuda')
    print(net)
    # [-1, 2048, 31, 31]
    summary(net, (3, 255, 255))
    print('*************\n')
    # [-1, 2048, 15, 15]
    summary(net, (3, 127, 127))

    # net = resnet50()
    # print(net)
    # net = net.cuda()
    #
    # var = torch.FloatTensor(1, 3, 127, 127).cuda()
    # # var = Variable(var)
    #
    # net(var)
    # print('*************')
    # var = torch.FloatTensor(1, 3, 255, 255).cuda()
    # # var = Variable(var)
    # net(var)
