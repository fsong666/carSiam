import json
import logging
import sys
import os
from os.path import join

import cv2
import numpy as np
from torch.utils.data import Dataset

from carsot.utils.bbox import center2corner, Center
from carsot.datasets.augmentation import Augmentation
from carsot.core.config import cfg

logger = logging.getLogger("global")

# setting opencv
pyv = sys.version[0]
if pyv[0] == '3':
    cv2.ocl.setUseOpenCL(False)


class SubDataset(object):
    """
    处理并加载各subset的json文件
    """

    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = os.path.join(cur_path, '../../', root)
        self.anno = os.path.join(cur_path, '../../', anno)
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            for track in meta_data[video]:
                frames = meta_data[video][track]
                # 获得只有数字int型的keys组成的list
                frames = list(map(int,
                                  filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
                # 为每个目标类对应的字典里添加int型frames索引
                meta_data[video][track]['frames'] = frames
                if len(frames) <= 0:
                    logger.warning("{}/{} has no frames".format(video, track))
                    del meta_data[video][track]

        for video in list(meta_data.keys()):
            if len(meta_data[video]) <= 0:
                logger.warning("{} has no tracks".format(video))
                del meta_data[video]

        self.labels = meta_data
        self.num = len(self.labels)
        self.num_use = self.num if self.num_use == -1 else self.num_use
        self.videos = list(meta_data.keys())  # 将keys转换为可index索引的list
        logger.info("{} json loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'  # e.g. 000000.00.x.jpg
        self.mask_format = "{}.{}.m.png"
        self.has_mask = name in ['COCO', 'YOUTUBEBB']
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
        """
        滤出bbox的w,h<=0的帧
        """

        meta_data_new = {}
        for video, tracks in meta_data.items():
            new_tracks = {}
            for trk, frames in tracks.items():
                new_frames = {}
                for frm, bbox in frames.items():
                    if not isinstance(bbox, dict):
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = bbox
                            w, h = x2 - x1, y2 - y1
                        else:
                            w, h = bbox
                        if w <= 0 or h <= 0:  # filter
                            continue
                    new_frames[frm] = bbox
                if len(new_frames) > 0:
                    new_tracks[trk] = new_frames
            if len(new_tracks) > 0:
                meta_data_new[video] = new_tracks
        return meta_data_new

    def log(self):
        logger.info("{} start-index {} select [{}/{}] path_format {}".format(
            self.name, self.start_idx, self.num_use,
            self.num, self.path_format))

    def shuffle(self):
        """
        len(lits) >= num_use,循环一次随机得到pick
        start_idx > 0 && len(lits) < num_use,循环多次得到pick
        各sub_set对应的pick范围是统一自然序列的分段随机
        """
        # lists是当前sub_set的实际的视频的绝对索引
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        # len(list)不是num_use整数倍时，会多出一倍，再截取
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        """
        par_crop.py 511和gen_json.py是平行的进程
        gen_json.py来自原始的数据集的注释.
        因为crop511有和原始集相同的目录结构，所以train.json可以调用crop511的数据
        """
        frame = "{:06d}".format(frame)  # 数字转str, 1 -> "000001"
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]  # 原始物体的bbox
        # print('imgpath: ', image_path)

        mask_path = join(self.root, video, self.mask_format.format(frame, track))

        return image_path, image_anno, mask_path

    def get_positive_pair(self, index):
        """
        index视频里随机选择一个跟踪目标，再从帧序列中随机选中一个模板帧，和范围内的随机搜索帧
        两帧都是000000.00.x.jpg图片 511x511x3
        """
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']

        # 随机选中一帧作为模板
        template_frame = np.random.randint(0, len(frames))
        # 确定从template_frame为中心向左搜索frame_range个(包括中心)，若超过0,则left为0
        left = max(template_frame - self.frame_range, 0)
        # 确定从template_frame为中心向右搜索frame_range个，若超过len,则right为len-1
        # 但切片取不到右边的终点，所以right+1
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        # 从search_range 随机选一个帧
        search_frame = np.random.choice(search_range)

        return self.get_image_anno(video_name, track, template_frame), \
               self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        """
        三次随机选中一帧
        :param index: 随机视频索引
        :return: 随机选中的跟踪目标的随机帧
        """
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]  # 得到video name string
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self, debug=False):
        super(TrkDataset, self).__init__()
        self.debug = debug
        # (255 -127) / 8 + 1 + 8 =? 25
        desired_size = (cfg.TRAIN.SEARCH_SIZE - cfg.TRAIN.EXEMPLAR_SIZE) / \
                       cfg.TRACK.STRIDE + 1 + cfg.TRAIN.BASE_SIZE
        if desired_size != cfg.TRAIN.OUTPUT_SIZE:
            raise Exception('size not match!')

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            # select coco if TrkDataset debug
            if self.debug and name != 'COCO':continue
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                name,
                subdata_cfg.ROOT,
                subdata_cfg.ANNO,
                subdata_cfg.FRAME_RANGE,
                subdata_cfg.NUM_USE,
                start
            )
            # 得到各自的分段随机pick
            start += sub_dataset.num
            self.num += sub_dataset.num_use

            sub_dataset.log()
            self.all_dataset.append(sub_dataset)

        # data augmentation
        self.template_aug = Augmentation(
            cfg.DATASET.TEMPLATE.SHIFT,
            cfg.DATASET.TEMPLATE.SCALE,
            cfg.DATASET.TEMPLATE.BLUR,
            cfg.DATASET.TEMPLATE.FLIP,
            cfg.DATASET.TEMPLATE.COLOR
        )
        self.search_aug = Augmentation(
            cfg.DATASET.SEARCH.SHIFT,
            cfg.DATASET.SEARCH.SCALE,
            cfg.DATASET.SEARCH.BLUR,
            cfg.DATASET.SEARCH.FLIP,
            cfg.DATASET.SEARCH.COLOR
        )
        videos_per_epoch = cfg.DATASET.VIDEOS_PER_EPOCH
        # self.num <= videos_per_epoch * epoch
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
        """
        将各自sub_set的随机重复的视频索引pick求和，
        再重复随机直到想要的总视频num数
        """
        pick = []
        m = 0
        while m < self.num:
            p = []
            for sub_dataset in self.all_dataset:
                sub_p = sub_dataset.pick
                p += sub_p
            np.random.shuffle(p)
            pick += p
            m = len(pick)
        logger.info("shuffle done!")
        logger.info("dataset length {}".format(self.num))
        return pick[:self.num]

    def _find_dataset(self, index):
        """
        返回sub_set内的实际有的视频并从0开始的绝对索引
        e.g. sub2的pick段是[3, 7, 5, 4, 6], start=3
        index = 7, sub2内的实际的video索引是 7 - 3 = 4, 7是整个set的索引
        """
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        """
        par_crp.py中，s_x与s_z近似同比例scale_z放大，
        返回{}.{}.x中的放大相同倍数的bbox
        :param image:{}.{}.x图片 511 x 511 x 3
        :param shape: 该图片对应原始未放大过的目标物体的bbox
        :return: 放大后的的bbox
        """
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w + h)  # margin p
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)  # 原始图里的小框  (w + 2p) × (h + 2p) = A
        scale_z = exemplar_size / s_z

        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        # print('bbox', shape)
        # print('bbox_center:',Center(cx, cy, w, h))
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index):
        """
        一次index只调用一个video中的一张模板帧所对应的pair,
        不是读取video的整个帧序列!!!
        为了保证video中的每张帧都能被使用，
        所以有NUM_USE重复video调用次数直到NUM_USE=video*frames
        但det,coco,是每个视频只有一张帧,所以无需重复,故NUM_USE=NUM
        """
        # print('-------\nidex:', index)
        index = self.pick[index]
        # print('pick idex:', index)
        dataset, index = self._find_dataset(index)
        # print('dataset idex:', index)

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        # neg = 0.2 > random
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()
        # get one dataset
        if neg:
            template = dataset.get_random_target(index)
            # 随机选择一个数据集中的随机一帧
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        # get image
        # template_image:(511, 511, 3) 用的是.x.jpg!!!不是.z.jpg 
        template_image = cv2.imread(template[0])
        if template_image is None:
            print('None in image_path:{}'.format(template[0]))

        # search_image: .x.jpg (511, 511, 3)
        search_image = cv2.imread(search[0])
        # print('template_image:', template_image.shape)
        # print('search_image:', search_image.shape)

        if dataset.has_mask and not neg:
            if cv2.imread(search[2], 0) is None:
                print('None in image_path:{} '.format(search[0]))
                return
            # print('has mask')
            search_mask = (cv2.imread(search[2], 0) > 0).astype(np.float32)
        else:
            # print('neg:', neg)
            search_mask = np.zeros(search_image.shape[:2], dtype=np.float32)

        # get bounding box  template[1] 原始物体的bbox
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])

        # augmentation template: (127, 127, 3)
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,  # 127
                                        gray=gray, debug=self.debug)
        # search: (255, 255, 3),  mask(255, 255)
        search, bbox, mask = self.search_aug(search_image,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,  # 255
                                       gray=gray, mask=search_mask, debug=self.debug)

        if self.debug:
            print('template_box:', template_box)
            print('search_box:', search_box)
            print('bbox_aug:', bbox)
            print('template:', template.shape)
            print('search:', search.shape)

            draw(template_image, template_box)
            draw(search_image, search_box)
            draw(template, _)
            draw(search, bbox)
            mask_debug = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            draw(mask_debug, bbox)

            cv2.imshow('template_image', template_image)
            cv2.imshow('search_image', search_image)
            cv2.imshow('template', template)
            cv2.imshow('search', search)
            cv2.imshow('search_mask', search_mask)
            cv2.imshow('mask', mask_debug)
            cv2.waitKey(0)

        cls = np.zeros((cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)

        # 取值[0,1]->[-1,1] for logistic_loss  (1, 255, 255) for 4d(bs,1, 255, 255) F.unfold
        mask = (np.expand_dims(mask, axis=0) > 0.5) * 2 - 1
        has_mask = np.zeros(1, dtype=np.int64)
        if dataset.has_mask and not neg:
            has_mask[0] = 1.
        return {
            'template': template,
            'search': search,
            'label_cls': cls,
            'bbox': np.array(bbox),
            'label_mask': np.array(mask, np.float32),
            'has_mask': has_mask
        }


def draw(image, box, color=(0, 255, 0)):
    if not box: return
    x1, y1, x2, y2 = map(lambda x: int(round(x)), box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color)


if __name__ == '__main__':
    train_dataset = TrkDataset(debug=True)
    train_dataset.__getitem__(0)

    cv2.destroyAllWindows()
