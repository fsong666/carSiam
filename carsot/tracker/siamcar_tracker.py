import numpy as np
import torch.nn.functional as F
import cv2

from carsot.core.config import cfg
from carsot.tracker.base_tracker import SiameseTracker
from carsot.utils.misc import bbox_clip

def pos_s_2_bbox(pos, s):
    """得到正方形框ｓ的四个定点的坐标"""
    return [pos[0] - s / 2, pos[1] - s / 2, pos[0] + s / 2, pos[1] + s / 2]


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
        self.debug = False

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
        self.size = np.array([bbox[2], bbox[3]])  # w, h

        # calculate z crop size
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)  # 0.5
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = round(np.sqrt(w_z * h_z))

        self.show_x_z_InputRegion(img, s_z)

        # calculate channle average
        self.channel_average = np.mean(img, axis=(0, 1))

        # get crop
        z_crop, _ = self.get_subwindow(img, self.center_pos,
                                       cfg.TRACK.EXEMPLAR_SIZE,
                                       s_z, self.channel_average)
        self.model.template(z_crop)

    def change(self, r):  # > 1
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
        penalty = np.exp(-(r_c * s_c - 1) * penalty_lk)  # penalty_lk= 0.1 or 0.04
        return penalty

    def accurate_location(self, max_r_up, max_c_up):
        """
        :return: 当前帧(255,255)里物体的bbox中心 与当前帧(255,255)的中心点坐标(127,127)的距离.
        当前帧(255,255)的中心是上一帧物体bbox中心点
        """
        dist = int((cfg.TRACK.INSTANCE_SIZE - (cfg.TRACK.SCORE_SIZE - 1) * 8) / 2)  # 31
        max_r_up += dist
        max_c_up += dist
        p_cool_s = np.array([max_r_up, max_c_up])
        disp = p_cool_s - (np.array([255, 255]) - 1.) / 2.  # - center([127., 127.])
        return disp

    def coarse_location(self, hp_cls_up, score_up, scale_score, lrtbs, img=None):
        """
        1. 在(193,193)的hp_cls_up全局中找到最大值的点(max_r_up_hp, max_c_up_hp) 青色 (Cyan),不是最终预测bbox!!
        2. 反缩小到(25,25),找到最大值点(max_r, max_c)对应的bbox 青色框
        3. 将该青色bbox旋转后得到黄色bbox
        4. 在score_up不再是hp_cls_up黄色bbox区域内的,局部求出最终的最大值点
        粗定位是求在hp_cls_up的最大值点
        精定位是求在考虑cen_up * cen_up的黄色区域的局部最大值点.
        :param hp_cls_up: 惩罚和高斯加权后的cls (193, 193)
        :param score_up: cls_up * cen_up
        :param scale_score: upsize / cfg.TRACK.SCORE_SIZE=8.64
        :param lrtbs: 原始tracker的预测的相对bbox值 [25, 25, 4]
        """
        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1
        max_r_up_hp, max_c_up_hp = np.unravel_index(hp_cls_up.argmax(), hp_cls_up.shape)  # (max_y, max_x)
        # cv2.circle(img, (max_c_up_hp + 31, max_r_up_hp + 31), 2, (255, 255, 0), 2)  # 垂直方向bbox中心
        # print('max_r_up_hp={} | max_c_up_hp= {}'.format(max_r_up_hp, max_c_up_hp))
        max_r = int(round(max_r_up_hp / scale_score))
        max_c = int(round(max_c_up_hp / scale_score))
        max_r = bbox_clip(max_r, 0, cfg.TRACK.SCORE_SIZE)
        max_c = bbox_clip(max_c, 0, cfg.TRACK.SCORE_SIZE)
        # print('max_r={} | max_c= {}'.format(max_r, max_c))
        # print('coarse_center in (25, 25)= ', (max_c, max_r))
        bbox_region = lrtbs[max_r, max_c, :]  # 0<bbox_region<255
        # print('bbox_region= ', bbox_region)
        min_bbox = int(cfg.TRACK.REGION_S * cfg.TRACK.EXEMPLAR_SIZE)  # 0.1*127=12.7
        max_bbox = int(cfg.TRACK.REGION_L * cfg.TRACK.EXEMPLAR_SIZE)  # 0.44*127=55.88
        # print('min_bbox={}, max_bbox={}'.format(min_bbox, max_bbox))

        # bbox < TRACK.EXEMPLAR_SIZE=127, 预测的bbox应在(127,127)模板框之内
        # 12 <= l <= 55
        # 将bbox_region的x轴顺时针旋转到垂直，将y轴逆时针旋转到水平方向,将预测框x,y值互相旋转了
        # [l,t,r,b] -> [t,l,b,r]
        l = bbox_clip(bbox_region[0], min_bbox, max_bbox)
        t = bbox_clip(bbox_region[1], min_bbox, max_bbox)
        r = bbox_clip(bbox_region[2], min_bbox, max_bbox)
        b = bbox_clip(bbox_region[3], min_bbox, max_bbox)
        pt1 = (int(max_c_up_hp - l + 31), int(max_r_up_hp - t + 31))
        pt2 = (int(max_c_up_hp + r + 31), int(max_r_up_hp + b + 31))
        # cv2.rectangle(img, pt1, pt2, (255, 255, 0), 1)  # 未旋转的bbox
        if self.debug:
            if l > max_r_up_hp or t > max_c_up_hp:
                print('hp_bbox_l=', l)
                print('hp_bbox_t=', t)
                cv2.rectangle(img, (31, 31), (31 + score_up.shape[1], 31 + score_up.shape[0]), (48, 48, 255), 1)
                # cv2.imshow('subwindow outer ', img)
                # cv2.waitKey(0)

        # 开始旋转处理
        l_region = int(min(max_r_up_hp, l))
        t_region = int(min(max_c_up_hp, t))
        r_region = int(min(upsize - max_r_up_hp, r))
        b_region = int(min(upsize - max_c_up_hp, b))
        # print('bbox_clip= ', (l_region, t_region, r_region, b_region))

        mask = np.zeros_like(score_up)
        # 截取旋转后的bbox区域为1
        mask[max_r_up_hp - l_region:max_r_up_hp + r_region + 1, max_c_up_hp - t_region:max_c_up_hp + b_region + 1] = 1

        # 画出旋转后的bbox
        x_min = max_c_up_hp - t_region + 31
        x_max = max_c_up_hp + b_region + 1 + 31
        y_min = max_r_up_hp - l_region + 31
        y_max = max_r_up_hp + r_region + 1 + 31
        # cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 255), 1)
        # (193, 193)
        # cv2.rectangle(img, (31, 31), (31 + mask.shape[1], 31 + mask.shape[0]), (125, 255, 0), 1)

        score_up = score_up * mask
        return score_up

    def getCenter(self, hp_cls_up, score_up, scale_score, lrtbs, img=None):
        # corse location
        score_up = self.coarse_location(hp_cls_up, score_up, scale_score, lrtbs, img)
        # accurate location
        max_r_up, max_c_up = np.unravel_index(score_up.argmax(), score_up.shape)
        # print('max_r_up={} | max_c_up= {}'.format(max_r_up, max_c_up))
        # cv2.circle(img, (max_c_up + 31, max_r_up + 31), 2, (0, 255, 255), 2)  # predicted center in (255,255)
        cv2.circle(img, (127, 127), 2, (0, 0, 255), 1)  # center of (255,255)
        # cv2.imshow('subwindow in track', img)

        disp = self.accurate_location(max_r_up, max_c_up)
        disp_ori = disp / self.scale_z  # 缩放到原图

        new_cx = disp_ori[1] + self.center_pos[0]
        new_cy = disp_ori[0] + self.center_pos[1]
        # return max_r_up, max_c_up, new_cx, new_cy
        return max_r_up, max_c_up, new_cx, new_cy, img

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

        or
        1.根据上一帧预测的bbox, 以此得到template和的search框. 从第二帧开始只获取search框
            giving bbox, template and search rectangle
        2.在当前帧的相同位置的search框里，获取缩放到255x255的sub_window,作为模型输入,得到新的预测bbox
            get 255sub_window and new bbox
        algorithm:
        bbox ＝ frame(last_bbox)
        last_bbox = bbox
        frame++
        """
        # print('--------')
        w_z = self.size[0] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        h_z = self.size[1] + cfg.TRACK.CONTEXT_AMOUNT * np.sum(self.size)
        s_z = np.sqrt(w_z * h_z)
        self.scale_z = cfg.TRACK.EXEMPLAR_SIZE / s_z  # from s_z scale to 127
        s_x = s_z * (cfg.TRACK.INSTANCE_SIZE / cfg.TRACK.EXEMPLAR_SIZE)

        # d_search = (instanc_size - exemplar_size) / 2  track时无需instanc_size, 直接在原图里track
        # pad = d_search / self.scale_z
        # s_x = s_z + 2 * pad

        self.show_x_z_InputRegion(img, s_z, s_x)

        # x_crop is tensor, x_img is np
        x_crop, x_img = self.get_subwindow(img, self.center_pos,
                                           cfg.TRACK.INSTANCE_SIZE,
                                           round(s_x), self.channel_average)
        outputs = self.model.track(x_crop)
        draw_anchor(x_img)

        cls = self._convert_cls(outputs['cls']).squeeze()  # [1, 2, 25, 25]-> [25, 25]
        cen = outputs['cen'].data.cpu().numpy().squeeze()  # [1, 1, 25, 25]-> [25, 25]
        lrtbs = outputs['loc'].data.cpu().numpy().squeeze()  # [1, 4, 25, 25]-> [4, 25, 25]
        if self.debug:
            print('cls.shape:= {} cen.shape= {} | lrtbs.shape={}'.format(
                cls.shape,
                cen.shape,
                lrtbs.shape))

        # cfg.TRACK.SCORE_SIZE == out_size= 25
        # upsize=193,是(255,255)图卷积时卷积核中心的轨迹范围
        upsize = (cfg.TRACK.SCORE_SIZE - 1) * cfg.TRACK.STRIDE + 1  # (25-1)*8+1=193
        # get penalty for output of subwindow
        penalty = self.cal_penalty(lrtbs, hp['penalty_k'])  # k_value
        p_cls = penalty * cls  # [25, 25]
        if cfg.TRACK.hanming:  # gauss
            # hp['window_lr'] == TRACK.WINDOW_INFLUENCE
            hp_cls = p_cls * (1 - hp['window_lr']) + self.window * hp['window_lr']
        else:
            hp_cls = p_cls
        best_idx = np.argmax(hp_cls)

        hp_cls_up = cv2.resize(hp_cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cls_up = cv2.resize(cls, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        cen_up = cv2.resize(cen, (upsize, upsize), interpolation=cv2.INTER_CUBIC)
        lrtbs = np.transpose(lrtbs, (1, 2, 0))  # [25, 25, 4]
        lrtbs_up = cv2.resize(lrtbs, (upsize, upsize), interpolation=cv2.INTER_CUBIC)  # [193, 193, 4]

        scale_score = upsize / cfg.TRACK.SCORE_SIZE  # 8.64
        #  cen_up downweight the scores of bounding boxes
        #  far from the center of an object
        score_up = cls_up * cen_up

        if self.debug:
            print('----after scale up to {}'.format(upsize))
            print('hp_cls_up.shape:= {} cls_up.shape= {} | lrtbs_up.shape={}'.format(
                hp_cls_up.shape,
                cls_up.shape,
                lrtbs_up.shape))

        # get center
        # max_r_up, max_c_up, new_cx, new_cy = self.getCenter(hp_cls_up, score_up, scale_score, lrtbs, x_img)
        max_r_up, max_c_up, new_cx, new_cy, subImg = self.getCenter(hp_cls_up, score_up, scale_score, lrtbs, x_img)

        # get w h in src img
        # lrtbs = (l,t,r,b)
        # w = l + r, h = t + b
        ave_w = (lrtbs_up[max_r_up, max_c_up, 0] + lrtbs_up[max_r_up, max_c_up, 2]) / self.scale_z
        ave_h = (lrtbs_up[max_r_up, max_c_up, 1] + lrtbs_up[max_r_up, max_c_up, 3]) / self.scale_z

        # smooth bbox with Interpolation learning rate
        scale_change = self.change(self.sz(ave_w, ave_h) /
                                   self.sz(self.size[0] * self.scale_z, self.size[1] * self.scale_z))
        ratio_change = self.change((self.size[0] / self.size[1]) / (ave_w / ave_h))
        # get penalty for source img　对预测边框的惩罚修剪
        penalty = np.exp(-(ratio_change * scale_change - 1) * hp['penalty_k'])
        # hp['lr'] Interpolation learning rate 预测边框的权重或者变化率
        # lr = 可能性*变化惩罚*占比
        lr = penalty * cls_up[max_r_up, max_c_up] * hp['lr']
        if self.debug:
            print('cls_up[max_r_up, max_c_up]=', cls_up[max_r_up, max_c_up])
            print('penalty=', penalty)
            print('lr=', lr)
        # 每个新的边框都是上次边框与新预测边框的加权和　lr是预测边框的权重
        new_width = lr * ave_w + (1 - lr) * self.size[0]
        new_height = lr * ave_h + (1 - lr) * self.size[1]

        # clip boundary
        cx = bbox_clip(new_cx, 0, img.shape[1])
        cy = bbox_clip(new_cy, 0, img.shape[0])
        width = bbox_clip(new_width, 0, img.shape[1])
        height = bbox_clip(new_height, 0, img.shape[0])

        draw_bbox_subImg(subImg, width, height, max_r_up, max_c_up, self.scale_z)

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

    def show_x_z_InputRegion(self, img, s_z, s_x=None):
        """
        根据上一帧的bbox在当前帧画出大小subwindow的框
        """
        z_rec = pos_s_2_bbox(self.center_pos, s_z)

        rect_bbox = np.trunc(np.array(z_rec)).astype(int)
        cv2.rectangle(img, (rect_bbox[0], rect_bbox[1]), (rect_bbox[2], rect_bbox[3]), (255, 0, 0),
                      thickness=1)
        if s_x:
            x_rec = pos_s_2_bbox(self.center_pos, s_x)
            rect_bbox = np.trunc(np.array(x_rec)).astype(int)
            cv2.rectangle(img, (rect_bbox[0], rect_bbox[1]), (rect_bbox[2], rect_bbox[3]), (0, 0, 255),
                          thickness=1)


def draw_anchor(img):
    # print all centers of slid windows
    ranges = np.arange(0, 25) * 8 + 31
    for v in ranges:
        for u in ranges:
            cv2.circle(img, (u, v), 1, (255, 50, 0), 1)


def draw_bbox_subImg(plane, width, height, max_c_up,  max_r_up, scale_z):
    # w, h in subwindow
    sub_cx, sub_cy = max_c_up + 31, max_r_up + 31
    w = int(bbox_clip(width * scale_z, 0, 255))
    h = int(bbox_clip(height * scale_z, 0, 255))
    # print((sub_cx, sub_cy, w, h))
    cv2.rectangle(plane,
                  (sub_cx - w//2, sub_cy - h//2),
                  (sub_cx + w//2, sub_cy + h//2),
                  (0, 255, 0), 1)
    # cv2.imshow('subwindow in track', plane)
    # cv2.imwrite('./trackImg.png', plane)

from carsot.utils.model_load import load_pretrain
from carsot.models.model_builder import ModelBuilder
from carsot.utils.bbox import get_axis_aligned_bbox
from os.path import join
from toolkit.datasets import DatasetFactory

if __name__ == '__main__':
    snapshot = '/home/sf/Documents/github_proj/carSiam/snapshot/general_model.pth'
    confg = '/home/sf/Documents/github_proj/carSiam/experiments/siamcar_r50/config.yaml'
    data_root = '/home/sf/Documents/github_proj/carSiam/testing_dataset/OTB50'
    cfg.merge_from_file(confg)

    model = ModelBuilder()
    model = load_pretrain(model, snapshot).cuda().eval()
    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    dataset = DatasetFactory.create_dataset(name='OTB50',
                                            dataset_root=data_root,
                                            load_img=False)
    params = [0.15, 0.1, 0.4]
    hp = {'lr': params[0], 'penalty_k': params[1], 'window_lr': params[2]}

    for v_idx, video in enumerate(dataset):
        if v_idx > 2: break
        pred_bboxes = []
        track_times = []
        for idx, (img, gt_bbox) in enumerate(video):
            # if idx > 8: break
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

                gt_bbox = list(map(int, gt_bbox))
                pred_bbox = list(map(int, pred_bbox))
                cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]),
                              (gt_bbox[0] + gt_bbox[2], gt_bbox[1] + gt_bbox[3]), (0, 255, 0), 2)

                cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]),
                              (pred_bbox[0] + pred_bbox[2], pred_bbox[1] + pred_bbox[3]), (0, 255, 255), 2)
                cx, cy, w, h = get_axis_aligned_bbox(np.array(pred_bbox))
                cv2.circle(img, (int(cx), int(cy)), 2, (0, 255, 255), 2)
                cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.imshow('img_output', img)
            cv2.waitKey(10)

    cv2.destroyAllWindows()
