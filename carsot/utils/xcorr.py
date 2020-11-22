import torch
import torch.nn.functional as F


def xcorr_slow(x, kernel):
    """for loop to calculate cross correlation, slow version
    out.bath=x.batch, out.channel=1
    会进行{卷积并求和]!!!}
    """
    batch = x.size()[0]
    out = []
    for i in range(batch):
        px = x[i]
        pk = kernel[i]  # px.size()[0] = channel
        px = px.view(1, px.size()[0], px.size()[1], px.size()[2])
        # px.size()[1] = channel
        pk = pk.view(-1, px.size()[1], pk.size()[1], pk.size()[2])
        po = F.conv2d(px, pk)
        #
        # print('px.size:{} corr pk.size:{} -> po.size:{}'.format(
        #     px.size(), pk.size(), po.size()))
        out.append(po)
    out = torch.cat(out, 0)
    return out


def xcorr_fast(x, kernel):
    """group conv2d to calculate cross correlation, fast version
   　批　== 组，　分批等效成分组

    groups=batch,each_group_c=in_channels//groups=x.channel with in_channels=batch*channel
    out_channels=groups
    总共groups个组,每组有channel张图片, 相应的有groups个,3d卷积核即channel个通道的卷积核.
    每组的多通道图与一个相同的通道数的卷积核进行{卷积并求和]!!!}得到一张单通道图.
    多组卷积核后得到共out_channels=groups张输出图.
    由bach分割后得到最终输出, out.bath=x.batch, out.channel=1

    xcorr_fast是xcorr_slow的替换加速版，运行过程相同.
    """
    batch = kernel.size()[0]  # x.size()[1] = channel
    pk = kernel.view(-1, x.size()[1], kernel.size()[2], kernel.size()[3])
    # view(1, -1, ...)将bach合并展成一个bach, 多个batch转成通道维度,分组卷积是分的channel
    # -1 = batch*channel
    px = x.view(1, -1, x.size()[2], x.size()[3])
    po = F.conv2d(px, pk, groups=batch)
    # print('px.size:{} corr pk.size:{} -> po.size:{}'.format(
    #     px.size(), pk.size(), po.size()))
    po = po.view(batch, -1, po.size()[2], po.size()[3])
    # print('po2.size:{}'.format(po.size()))
    return po


def xcorr_depthwise(x, kernel):
    """depthwise cross correlation
    groups=batch*channel,each_group_c=in_channels//groups= 1 with in_channels=batch*channel
    out_channels=groups
    每组只有一张图片，所以每张单通道图片与单通道卷积核进行卷积,输出一个out_channel的图片, 总共得到out_channels张图片
    输出的图片{不再是进行求和!!!}
    由bach分割后得到最终输出，输入输出的通道数不变即 out.channel == x.channel
    """
    batch = kernel.size(0)
    channel = kernel.size(1)    # 256
    x = x.view(1, batch*channel, x.size(2), x.size(3))
    kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
    out = F.conv2d(x, kernel, groups=batch*channel)
    # print('px.size:{} corr pk.size:{} -> out.size:{}'.format(
    #     x.size(), kernel.size(), out.size()))
    out = out.view(batch, channel, out.size(2), out.size(3))
    # print('out2.size:{}'.format(out.size()))
    return out


if __name__ == '__main__':
    xx = 31
    zz = 7
    bs = 2
    c = 256
    x = torch.rand((bs, c, xx, xx)).to('cuda')
    z = torch.rand((bs, c, zz, zz)).to('cuda')
    print('x={} | z={} '.format(x.size(), z.size()))

    print('*****xcorr_slow******')
    out = xcorr_slow(x, z)  # [-1, 1, 25, 25]
    print('out.shape:{} \nout=\n{} '.format(out.shape, out.shape))

    print('*****xcorr_fast******')
    out = xcorr_fast(x, z)  # [-1, 1, 25, 25]
    print('out.shape:{} \nout=\n{} '.format(out.shape, out.shape))

    print('*****xcorr_depthwise******')
    out = xcorr_depthwise(x, z)  # [-1, 256, 25, 25])
    print('out.shape:{} \nout=\n{} '.format(out.shape, out.shape))