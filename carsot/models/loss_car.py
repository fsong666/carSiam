"""
This file contains specific functions for computing losses of SiamCAR
file
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
import logging
from torch.autograd import Variable

logger = logging.getLogger('global')

INF = 100000000


def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)  # only torch.int64(long)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    """
    pred[:,0] = 0 = 背景
    pred[:,1] = 1 = 物体
    正反两类的损失都算了
    """
    # [bs, 1, 25, 25, 2] -> [bs*625, 2]
    pred = pred.view(-1, 2)
    label = label.view(-1)
    # 分别得到一维数组label的等与0和等于1的索引
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


class IOULoss(nn.Module):
    """
    x的625个grid的中心点也可以输出
    weight= centerness_targets
    """
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        losses = -torch.log((area_intersect + 1.0) / (area_union + 1.0))

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class SiamCARLossComputation(object):
    """
    This class computes the SiamCAR losses.
    """

    def __init__(self, cfg):
        self.box_reg_loss_func = IOULoss()
        #  If (x; y) is located in the background, the
        # value of C(i; j) is set to 0.
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        # targets shape [pos] 是[0, 1]的浮点值,所以用回归的loss MSELoss
        # self.centerness_loss_func = nn.MSELoss()    # target center-ness ranges from 0 to 1
        self.cfg = cfg

    def prepare_targets(self, points, labels, gt_bbox):

        labels, reg_targets = self.compute_targets_for_locations(
            points, labels, gt_bbox
        )

        return labels, reg_targets

    def compute_targets_for_locations(self, locations, labels, gt_bbox, debug=False):
        """
        输入的x图片有25*25=625个gird点, 代表每次卷积时卷积核的中心坐标,卷积stride=8,
        共625次卷积(cross-correlation)
        (28,28) = cnnSize(in_size=(255,255), kernel_size=32, stride=8, padding=0)
        (28,28)->(25,25)
        求得25x25相似度图的一个相似度值,通过该相似度值可以进一步提取目标物体的定位即cls,
        以及grid中心点相对于bbox边界的四个值[l, t, r, b] 和　相对于bbox中心的偏离值.

        l, t, r, b 是每个搜索grid的中心点的坐标，相对于gt_bbox对角点的坐标值.
        gt_bbox给定，四个相对值就是常数给定的目标值.
        作用：通过x的任意一个grid中心点的[l, t, r, b],画出目标bbox!!!
        不是用来画出grid的边界!

        labels会缩小一个维度,将二维图转成一列一维表示.
        labels列维度表示图的像素值，也对应输入ｘ[3, 255, 255]里的TRACK.STRID=8的一个8x8的小方格.
        每个方格代表一次模板和x的卷积，卷积结果对应输出特征图的一个像素值.
        """
        # reg_targets = []
        # xs一维数组
        xs, ys = locations[:, 0], locations[:, 1]

        bboxes = gt_bbox
        # [bs, 25, 25] -> [25**2=625, bs=32], 每一列代表一张图
        labels = labels.view(self.cfg.TRAIN.OUTPUT_SIZE ** 2, -1)

        # xs[:, None]是xs的列向量, None 增加一个维度
        # [625, 1] - [1, bs] = [625, bs]
        l = xs[:, None] - bboxes[:, 0][None].float()
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        if debug:
            print('xs=', xs.shape)
            print('xs=', xs)
            print('gt_bbox.shape=', bboxes.shape)
            print('l.shape=', l.shape)

        # [625, 32, 4]: 625个方格, 32batch, 4个相对值
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

        s1 = reg_targets_per_im[:, :, 0] > 0.6 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
        s2 = reg_targets_per_im[:, :, 2] > 0.6 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
        s3 = reg_targets_per_im[:, :, 1] > 0.6 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
        s4 = reg_targets_per_im[:, :, 3] > 0.6 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
        is_in_boxes = s1 * s2 * s3 * s4  # [625, 32] 对应列索引一张图的像素点，32代表对应batch索引
        pos = np.where(is_in_boxes.cpu() == 1)
        labels[pos] = 1  # [25**2=625, 32]

        # [bs, 625]  [bs, 625, 4]
        return labels.permute(1, 0).contiguous(), reg_targets_per_im.permute(1, 0, 2).contiguous()

    def compute_centerness_targets(self, reg_targets):
        """
        center-ness can downweight the scores of bounding boxes far from the center
        of an object
        The center-ness ranges from 0 to 1

        ｘ的cls为正例的点相对于bbox中心的偏离程度.
        :param reg_targets: 四个相对ｂbox的目标值
        :return: 输入x里pos个点的相对与bbox中心的评测值,也是输入给定后的待预测的target
        bbox的中心的相对值是1，其他都小于1. shape: [pos]
        其pos里的分布类似以gt_bbox中心的高斯分布
        """
        left_right = reg_targets[:, [0, 2]]  # [-1, 2]
        top_bottom = reg_targets[:, [1, 3]]
        # .min(dim=-1 返回的是两个元素的tuple,第一个元素代表最小，第二个代表其索引
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                     (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness,
                 labels, reg_targets, has_mask=None, pred_mask=None, label_mask=None, debug=False):
        """
        Arguments:
            locations (list[BoxList]): center of sliding window [625, 2]
            box_cls (list[Tensor])  pred_cls
            box_regression (list[Tensor])  pred_reg
            centerness (list[Tensor]) pred_center
            labels: zeros [25,25]
            reg_targets: target bbox

        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)

        label生成过程:
        1.生成(25,25)的ltrb
        2.ltrb中全部为正且在gz_bbox中心区域的点为正例点pos,其余为0,得到(25,25)label_cls
        3.在pos区域内生成相对于gt_bbox中心的偏离值centerness_targets, shape:[pos]

        loss过程:
        1. 通过ltrb[pos]的IOU，得到reg_loss, 只计算pos区域, 也只对pos区域进行反向优化
        2. pred_cls与label_cls,　得到clc_loss, 针对(25,25)
        3. centerness_flatten[pos] 与 centerness_targets, 逻辑回归得到cen_loss, 只对pos区域计算
        各loss相互独立
        """
        # generate label
        label_cls, reg_targets = self.prepare_targets(locations, labels, reg_targets)

        # [bs, 4, 25, 25]-> [bs, 25, 25, 4] -> [bs*625, 4]
        box_regression_flatten = (box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4))
        labels_flatten = (label_cls.view(-1))  # [bs, 625] -> [bs*625]
        reg_targets_flatten = (reg_targets.view(-1, 4))  # [bs, 625, 4] -> [bs*625, 4]
        centerness_flatten = (centerness.view(-1))  # [bs, 1, 25, 25] -> [bs*625]

        # 一维数组labels[bs * 625]中非零值的索引
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze()
        if debug: print('pos_inds=', pos_inds.numel())

        box_regression_flatten = box_regression_flatten[pos_inds]
        reg_targets_flatten = reg_targets_flatten[pos_inds]
        centerness_flatten = centerness_flatten[pos_inds]
        cls_loss = select_cross_entropy_loss(box_cls, labels_flatten)

        mask_loss = None
        if cfg.MASK.MASK:
            mask_weight = has_mask * label_cls
            if debug: print('mask_weight:', mask_weight.shape)
            mask_loss = select_mask_logistic_loss(pred_mask, label_mask, mask_weight)

        if pos_inds.numel() > 0:

            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)

            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )
            # print('centerness_targets=\n', centerness_targets)
            # print('\n------------------')
            # print('cen_loss=', centerness_loss)
            # test_BCE(centerness_flatten, centerness_targets)
        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss, mask_loss


def make_siamcar_loss_evaluator(cfg):
    loss_evaluator = SiamCARLossComputation(cfg)
    return loss_evaluator


def select_mask_logistic_loss(p_m, mask, weight, o_sz=63, g_sz=127, debug=False):
    weight = weight.view(-1)  # 25*25*bs
    pos = Variable(weight.data.eq(1).nonzero().squeeze())  # grid点含有anchors的cls等于1,视为pos
    if pos.nelement() == 0:
        return p_m.sum() * 0
    if debug:
        print('------mask_loss----')
        print('pos=', pos.shape)
        print('p_m=', p_m.shape)

    if len(p_m.shape) == 4:
        # [bs, 63*63=3696, 25, 25]->[bs, 25, 25, 3696]->[25*25*bs,1,63,63]
        p_m = p_m.permute(0, 2, 3, 1).contiguous().view(-1, 1, o_sz, o_sz)
        p_m_check = p_m
        p_m = torch.index_select(p_m, 0, pos)  # [len(pos), 1, 63, 63]
        # p_m = nn.UpsamplingBilinear2d(size=[g_sz, g_sz])(p_m)   # [len(pos), 1, 127, 127])
        p_m = F.interpolate(p_m, size=(g_sz, g_sz), mode='bilinear', align_corners=True)
        p_m = p_m.view(-1, g_sz * g_sz)  # [len(pos),127*127])
    else:
        p_m_check = p_m
        p_m = torch.index_select(p_m, 0, pos)

    # only 4-D input tensors supported.
    # padding=32, stride=8 for out_size(25, 25)
    # padding=0, stride=8 for out_size(17, 17)
    if cfg.REFINE.REFINE:
        mask_uf = F.unfold(mask, (g_sz, g_sz), padding=0, stride=8)  # [B, C_kh_kw, L]= [bs, 16129=127*127, 625=25*25]
    else:
        mask_uf = F.unfold(mask, (g_sz, g_sz), padding=32, stride=8)  # [B, C_kh_kw, L]= [bs, 16129=127*127, 625=25*25]
    if debug: print('unfold mask_uf=', mask_uf.shape)
    # [B, C_kh_kw, L] ->[B*L*C, kH*kW]= [bs*625, 16129]  target_mask的c一直是1
    mask_uf = torch.transpose(mask_uf, 1, 2).contiguous().view(-1, g_sz * g_sz)
    if debug: print('mask_uf={} -- p_m_check={}'.format(mask_uf.shape, p_m_check.shape))
    assert mask_uf.shape[0] == p_m_check.shape[0]

    mask_uf = torch.index_select(mask_uf, 0, pos)  # [len(pos),127*127])
    if debug:
        print('mask_uf=', mask_uf.shape)
        print('p_m=', p_m.shape)
    loss = F.soft_margin_loss(p_m, mask_uf)
    return loss


from carsot.core.config import cfg

def test_BCE(x=None, target=None):
    # x = torch.tensor([0.4859, 0.4891, 0.5124, 0.5517, 0.5466, 0.6050, 0.6091, 0.6281], device='cuda:0', requires_grad=True)
    # target= torch.tensor([0.6492, 0.6815, 0.7760, 0.8147, 0.7364, 0.7731, 0.6140, 0.6446], device='cuda:0', requires_grad=True)
    # print('x=\n', x)
    # print('traget=\n', target)
    print('bce_loss=', F.binary_cross_entropy_with_logits(x, target))

if __name__ == '__main__':
    # torch.manual_seed(10)
    # locations = torch.randint(0, 225, (625, 2))
    # labels = torch.zeros((32, cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE))
    # gt_bbox = torch.randint(0, 255, (32, 4))
    # evaluator = SiamCARLossComputation(cfg)
    # evaluator.compute_targets_for_locations(locations=locations, labels=labels, gt_bbox=gt_bbox)

    # b = 4
    # pred = torch.randint(0, 255, (b, 2)).float()
    # print('pred:', pred)
    # label = torch.zeros(b)
    # s = torch.tensor([0, 1, 1, 0])
    # out = F.nll_loss(pred, s)
    # print(out)
    test_BCE()

