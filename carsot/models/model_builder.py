import torch
import torch.nn as nn
import torch.nn.functional as F

from carsot.core.config import cfg
from carsot.models.loss_car import make_siamcar_loss_evaluator
from carsot.models.backbone import get_backbone
from carsot.models.head.car_head import CARHead
from carsot.models.neck import get_neck
from ..utils.location_grid import compute_locations
from carsot.utils.xcorr import xcorr_depthwise
import logging
logger = logging.getLogger('global')


class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer AdjustAllLayer to [256, 256, 256]
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build car head
        self.car_head = CARHead(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_siamcar_loss_evaluator(cfg)

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf[0],self.zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],self.zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)

        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()    # [-1, 3, 127, 127]
        search = data['search'].cuda()        # [-1, 3, 255, 255]
        label_cls = data['label_cls'].cuda()  # [-1, 25, 25]
        label_loc = data['bbox'].cuda()       # [-1, 4]
        # logger.warning('label_cls.size:{}'.format(label_cls.size()))

        # get feature
        zf = self.backbone(template)  # [layer[-1], -1, 2048, 15, 15]
        xf = self.backbone(search)    # [layer[-1], -1, 2048, 31, 31]
        # logger.warning('********resnet********')
        # logger.warning('xf.size:{}'.format(xf[2].size()))
        # logger.warning('zf.size:{}'.format(zf[2].size()))
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)        # [layer, -1, 256, 31, 31]
            xf = self.neck(xf)        # [layer, -1, 256,  7,  7]

        # logger.warning('********neck********')
        # logger.warning('xf.size:{}'.format(xf[2].size()))
        # logger.warning('zf.size:{}'.format(zf[2].size()))

        # 后面三层各自进行z与x的卷积,然后cat
        features = self.xcorr_depthwise(xf[0],zf[0])  # [-1, 256, 25, 25]
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],zf[i+1])
            features = torch.cat([features,features_new],1)
        # [-1, 256*3, 25, 25]
        features = self.down(features)  # [-1, 256, 25, 25]
        # logger.warning('features.size:{}'.format(features.size()))

        cls, loc, cen = self.car_head(features)  # [-1, -1, 25, 25]
        # logger.warning('********car_head********')
        # logger.warning('cls.size:{}'.format(cls.size()))
        # logger.warning('loc.size:{}'.format(loc.size()))
        # logger.warning('cen.size:{}'.format(cen.size()))

        locations = compute_locations(cls, cfg.TRACK.STRIDE)  # [625, 2]
        cls = self.log_softmax(cls)  # [-1, 1, 25, 25, 2]

        # loss_evaluator.__call__
        # logger.warning('********loss_evaluator********')
        cls_loss, loc_loss, cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        # get loss WEIGHT=[1, 3, 1]
        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs


