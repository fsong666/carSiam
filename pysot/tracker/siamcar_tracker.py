import numpy as np
import torch.nn.functional as F
import cv2

from pysot.core.config import cfg
from pysot.tracker.base_tracker import SiameseTracker
from pysot.utils.misc import bbox_clip


def pos_s_2_bbox(pos, s):
    """得到正方形框ｓ的四个定点的坐标"""
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]

def center2corner(center):
    """ convert (cx, cy, w, h) to (x1, y1, x2, y2)
    Args:
        center: Center or np.array (4 * N)
    Return:
        center or np.array (4 * N)
    """
    x, y, w, h = center[0], center[1], center[2], center[3]
    x1 = x - w * 0.5
    y1 = y - h * 0.5
    x2 = x + w * 0.5
    y2 = y + h * 0.5
    return x1, y1, x2, y2


class SiamCARTracker(SiameseTracker):
    def __init__(self, model, cfg):
        super(SiamCARTracker, self).__init__()
        hanning = np.hanning(cfg.SCORE_SIZE)
        self.window = np.outer(hanning, hanning)
        self.model = model
        self.model.eval()

    def _convert_cls(self, cls):
        cls = F.softmax(cls[:, :, :, :], dim=1).data[:, 1, :, :].cpu().numpy()
        return cls

    def init(self, img, bbox):
        """
        args:
            img(np.ndarray): BGR image
            bbox: (x, y, w, h) bbox  gt_bbox_
        """
        self.center_pos = np.array([bbox[0] + (bbox[2] - 1) / 2,
                                    bbox[1] + (bbox[3] - 1) / 2])
        self.size = np.array([bbox[2], bbox[3]]) # w, h

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)  # 0.5
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.EXEMPLAR_SIZE,
                                    s_z, self.channel_average)
        self.model.template(z_crop)

    def change(self, r):
        return np.maximum(r, 1. / r)

    def sz(self, w, h):
        pad = (w + h) * 0.5
        return np.sqrt((w + pad) * (h + pad))

    def cal_penalty(self, lrtbs, penalty_lk):
        bboxes_w = lrtbs[0, :, :] + lrtbs[2, :, :]
        bboxes_h = lrtbs[1, :, :] + lrtbs[3, :, :]
        s_c = self.change(
            self.sz(bboxes_w, bboxes_h) / self.sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (bboxes_w / bboxes_h))
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)
        return penalty

    def accurate_location(self, max_r_up, max_c_up):
        dist = int((cfg.TRACK.INSTANCE_SIZE - (cfg.TRACK.SCORE_SIZE - 1) * 8) / 2)
        max_r_up += dist
        max_c_up += dist
        p_cool_s = np.array([max_r_up, max_c_up])
        disp = p_cool_s - (np.array([255, 255]) - 1.) / 2.
        return disp

    def coarse_location(self, hp_cls_up, score_up, scale_score, lrtbs):
        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        max_r_up_hp, max_c_up_hp = np.unravel_index(hp_cls_up.argmax(), hp_cls_up.shape)
        max_r = int(round(max_r_up_hp / scale_score))
        max_c = int(round(max_c_up_hp / scale_score))
        max_r = bbox_clip(max_r, 0, cfg.TRACK.SCORE_SIZE)
        max_c = bbox_clip(max_c, 0, cfg.TRACK.SCORE_SIZE)
        bbox_region = lrtbs[max_r, max_c, :]
        min_bbox = int(cfg.TRACK.REGION_S * cfg.TRACK.EXEMPLAR_SIZE)
        max_bbox = int(cfg.TRACK.REGION_L * cfg.TRACK.EXEMPLAR_SIZE)
        l_region = int(min(max_r_up_hp, bbox_clip(bbox_region[0], min_bbox, max_bbox)))
        t_region = int(min(max_c_up_hp, bbox_clip(bbox_region[1], min_bbox, max_bbox)))

        r_region = int(min(upsize - max_r_up_hp, bbox_clip(bbox_region[2], min_bbox, max_bbox)))
        b_region = int(min(upsize - max_c_up_hp, bbox_clip(bbox_region[3], min_bbox, max_bbox)))
        mask = np.zeros_like(score_up)
        mask[max_r_up_hp - l_region:max_r_up_hp + r_region + 1, max_c_up_hp - t_region:max_c_up_hp + b_region + 1] = 1
        score_up = score_up * mask
        return score_up

    def getCenter(self, hp_cls_up, score_up, scale_score, lrtbs):
        # corse location
        score_up = self.coarse_location(hp_cls_up, score_up, scale_score, lrtbs)
        # accurate location
        max_r_up, max_c_up = np.unravel_index(score_up.argmax(), score_up.shape)
        disp = self.accurate_location(max_r_up, max_c_up)
        disp_ori = disp / self.scale_z
        new_cx = disp_ori[1] + self.center_pos[0]
        new_cy = disp_ori[0] + self.center_pos[1]
        return max_r_up, max_c_up, new_cx, new_cy

    def track(self, img, hp):
        """
        args:
            img(np.ndarray): BGR image
        return:
            bbox(list):[x, y, width, height]
        用当前的已知的预测框的self.center_pos, self.size，在下一张输入图img里用相同的预测框坐标center_pos，size来
        截取放大得到model要求的search subwindow 255x255即crop_bbox,
        然后输入模型预测得到该img的预测框self.center_pos, self.size
        重复以上过程
        """

        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        # d_search = (instanc_size - exemplar_size) / 2  track时无需instanc_size, 直接在原图里track
        # pad = d_search / self.scale_z
        # s_x = s_z + 2 * pad

        z_rec = pos_s_2_bbox(self.center_pos, s_z)
        x_rec = pos_s_2_bbox(self.center_pos, s_x)
        rect_bbox = np.trunc(np.array(z_rec)).astype(int)
        rect_bbox = cv2.rectangle(img, (rect_bbox[0], rect_bbox[1]), (rect_bbox[2], rect_bbox[3]), (255, 0, 0),
                                  thickness=2)
        rect_bbox = np.trunc(np.array(x_rec)).astype(int)
        rect_bbox = cv2.rectangle(img, (rect_bbox[0], rect_bbox[1]), (rect_bbox[2], rect_bbox[3]), (0, 0, 255),
                                  thickness=2)

        x_crop = self.get_subwindow(img, self.center_pos,
                                    cfg.TRACK.INSTANCE_SIZE,
                                    round(s_x), self.channel_average)

        outputs = self.model.track(x_crop)
        cls = self._convert_cls(outputs['cls']).squeeze()
        cen = outputs['cen'].data.cpu().numpy().squeeze()
        lrtbs = outputs['loc'].data.cpu().numpy().squeeze()

        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        penalty = self.cal_penalty(lrtbs, hp['penalty_k'])
        p_cls = penalty * cls
        if cfg.TRACK.hanming:
            hp_cls = p_cls * (1 - hp['window_lr']) + self.window * hp['window_lr']
        else:
            hp_cls = p_cls

        hp_cls_up = cv2.resize(hp_cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cen_up = cv2.resize(cen, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        lrtbs = np.transpose(lrtbs, (1, 2, 0))
        lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)

        scale_score = upsize / cfg.TRACK.SCORE_SIZE
        score_up = cls_up * cen_up

        # get center
        max_r_up, max_c_up, new_cx, new_cy = self.getCenter(hp_cls_up, score_up, scale_score, lrtbs)

        # get w h
        ave_w = (lrtbs_up[max_r_up, max_c_up, 0] + lrtbs_up[max_r_up, max_c_up, 2]) / self.scale_z
        ave_h = (lrtbs_up[max_r_up, max_c_up, 1] + lrtbs_up[max_r_up, max_c_up, 3]) / self.scale_z

        s_c = self.change(self.sz(ave_w, ave_h) / self.sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z))
        r_c = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        penalty = np.exp(-(r_c * s_c - 1) * hp['penalty_k'])
        lr = penalty * cls_up[max_r_up, max_c_up] * hp['lr']
        new_width = lr * ave_w + (1 - lr) * self.size[0]
        new_height = lr * ave_h + (1 - lr) * self.size[1]

        # clip boundary
        cx = bbox_clip(new_cx, 0, img.shape[1])
        cy = bbox_clip(new_cy, 0, img.shape[0])
        width = bbox_clip(new_width, 0, img.shape[1])
        height = bbox_clip(new_height, 0, img.shape[0])

        # udpate state
        self.center_pos = np.array([cx, cy])
        self.size = np.array([width, height])
        bbox = [cx - width / 2,
                cy - height / 2,
                width,
                height]

        return {
            'bbox': bbox,
        }


from pysot.utils.model_load import load_pretrain
from pysot.models.model_builder import ModelBuilder
from pysot.utils.bbox import get_axis_aligned_bbox
from os.path import join
from toolkit.datasets import DatasetFactory

if __name__ == '__main__':
    snapshot = '/home/sf/Documents/github_proj/SiamCAR/snapshot/checkpoint_e13.pth'
    confg = '/home/sf/Documents/github_proj/SiamCAR/experiments/siamcar_r50/config.yaml'
    data_root = '/home/sf/Documents/github_proj/SiamCAR/testing_dataset/VOT2018'
    cfg.merge_from_file(confg)

    model = ModelBuilder()
    model = load_pretrain(model, snapshot).cuda().eval()
    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    dataset = DatasetFactory.create_dataset(name='VOT2018',
                                            dataset_root=data_root,
                                            load_img=False)
    params = [0.15, 0.1, 0.4]
    hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}

    for v_idx, video in enumerate(dataset):
        if v_idx > 0: break
        pred_bboxes = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            if idx > 2: break
            tic = cv2.getTickCount()
            if idx == 0:
                cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
                gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]  # 得到bbox左上角的点
                tracker.init(img, gt_bbox_)
                pred_bbox = gt_bbox_  # 第一张图作为模板，不输入模型进行预测
                pred_bboxes.append(pred_bbox)

                gt = get_axis_aligned_bbox(np.array(gt_bbox))
                gt = center2corner(gt)
                rect_bbox = np.trunc(np.array(gt)).astype(int)  # 截取整数位,丢弃小数位
                rect_bbox = cv2.rectangle(img, (rect_bbox[0], rect_bbox[1]), (rect_bbox[2], rect_bbox[3]), (0, 255, 0),
                                          thickness=2)
                print('gt_bbox', gt_bbox)
                cv2.imshow('init_img_input', img)
            else:  # 从第二张开始模型输出
                outputs = tracker.track(img, hp)
                pred_bbox = outputs['bbox']
                pred_bboxes.append(pred_bbox)
                cv2.imshow('img_input', img)
            cv2.waitKey(0)


    cv2.destroyAllWindows()
