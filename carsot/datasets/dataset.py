import json
import logging
import sys
import os
from collections import namedtuple

Corner = namedtuple('Corner', 'x1 y1 x2 y2')

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
    def __init__(self, name, root, anno, frame_range, num_use, start_idx):
        cur_path = os.path.dirname(os.path.realpath(__file__))
        self.name = name
        self.root = root
        self.anno = os.path.join(cur_path, '../../', anno)
        self.frame_range = frame_range
        self.num_use = num_use
        self.start_idx = start_idx
        logger.info("loading " + name)
        with open(self.anno, 'r') as f:
            meta_data = json.load(f)
            meta_data = self._filter_zero(meta_data)

        for video in list(meta_data.keys()):
            # print('------------\n:', meta_data[video])
            for track in meta_data[video]:
                frames = meta_data[video][track]
                frames = list(map(int,
                                  filter(lambda x: x.isdigit(), frames.keys())))
                frames.sort()
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
        self.videos = list(meta_data.keys())
        logger.info("{} loaded".format(self.name))
        self.path_format = '{}.{}.{}.jpg'
        self.pick = self.shuffle()

    def _filter_zero(self, meta_data):
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
                        if w <= 0 or h <= 0:  # remove w h <= 0
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
        lists = list(range(self.start_idx, self.start_idx + self.num))
        pick = []
        while len(pick) < self.num_use:
            np.random.shuffle(lists)
            pick += lists
        return pick[:self.num_use]

    def get_image_anno(self, video, track, frame):
        frame = "{:06d}".format(frame)
        image_path = os.path.join(self.root, video,
                                  self.path_format.format(frame, track, 'x'))
        image_anno = self.labels[video][track][frame]
        return image_path, image_anno

    def get_positive_pair(self, index):
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]

        frames = track_info['frames']
        template_frame = np.random.randint(0, len(frames))
        left = max(template_frame - self.frame_range, 0)
        right = min(template_frame + self.frame_range, len(frames) - 1) + 1
        search_range = frames[left:right]
        template_frame = frames[template_frame]
        search_frame = np.random.choice(search_range)
        return self.get_image_anno(video_name, track, template_frame), \
               self.get_image_anno(video_name, track, search_frame)

    def get_random_target(self, index=-1):
        if index == -1:
            index = np.random.randint(0, self.num)
        video_name = self.videos[index]
        video = self.labels[video_name]
        track = np.random.choice(list(video.keys()))
        track_info = video[track]
        frames = track_info['frames']
        frame = np.random.choice(frames)
        return self.get_image_anno(video_name, track, frame)

    def __len__(self):
        return self.num


class TrkDataset(Dataset):
    def __init__(self, ):
        super(TrkDataset, self).__init__()

        # create sub dataset
        self.all_dataset = []
        start = 0
        self.num = 0
        for name in cfg.DATASET.NAMES:
            subdata_cfg = getattr(cfg.DATASET, name)
            sub_dataset = SubDataset(
                name,
                subdata_cfg.ROOT,
                subdata_cfg.ANNO,
                subdata_cfg.FRAME_RANGE,
                subdata_cfg.NUM_USE,
                start
            )
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
        self.num = videos_per_epoch if videos_per_epoch > 0 else self.num
        self.num *= cfg.TRAIN.EPOCH
        self.pick = self.shuffle()

    def shuffle(self):
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
        for dataset in self.all_dataset:
            if dataset.start_idx + dataset.num > index:
                return dataset, index - dataset.start_idx

    def _get_bbox(self, image, shape):
        imh, imw = image.shape[:2]
        if len(shape) == 4:
            w, h = shape[2] - shape[0], shape[3] - shape[1]
        else:
            w, h = shape
        context_amount = 0.5
        exemplar_size = cfg.TRAIN.EXEMPLAR_SIZE
        wc_z = w + context_amount * (w + h)
        hc_z = h + context_amount * (w + h)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = exemplar_size / s_z
        w = w * scale_z
        h = h * scale_z
        cx, cy = imw // 2, imh // 2
        bbox = center2corner(Center(cx, cy, w, h))
        return bbox

    def __len__(self):
        return self.num

    def __getitem__(self, index, debug=False):
        index = self.pick[index]
        dataset, index = self._find_dataset(index)

        gray = cfg.DATASET.GRAY and cfg.DATASET.GRAY > np.random.random()
        neg = cfg.DATASET.NEG and cfg.DATASET.NEG > np.random.random()

        # get one dataset
        if neg:
            template = dataset.get_random_target(index)
            search = np.random.choice(self.all_dataset).get_random_target()
        else:
            template, search = dataset.get_positive_pair(index)

        # get image
        template_image = cv2.imread(template[0])
        search_image = cv2.imread(search[0])
        if template_image is None:
            print('error image:', template[0])

        # get bounding box
        template_box = self._get_bbox(template_image, template[1])
        search_box = self._get_bbox(search_image, search[1])

        # augmentation
        template, _ = self.template_aug(template_image,
                                        template_box,
                                        cfg.TRAIN.EXEMPLAR_SIZE,
                                        gray=gray)

        search, bbox = self.search_aug(search_image,
                                       search_box,
                                       cfg.TRAIN.SEARCH_SIZE,
                                       gray=gray)
        if debug:
            cv2.imshow('template_input', template_image)
            cv2.imshow('search_input', search_image)
            cv2.imshow('template_aug', template)
            cv2.imshow('search_aug', search)
            cv2.waitKey(0)

        # int64 used as label in F.nll_loss(, label), label support only int64(long)
        cls = np.zeros((cfg.TRAIN.OUTPUT_SIZE, cfg.TRAIN.OUTPUT_SIZE), dtype=np.int64)
        template = template.transpose((2, 0, 1)).astype(np.float32)
        search = search.transpose((2, 0, 1)).astype(np.float32)

        return {
            'template': template,
            'search': search,
            'label_cls': cls,  # label_cls=torch.Size([32, 25, 25])
            'bbox': np.array([bbox.x1, bbox.y1, bbox.x2, bbox.y2])
        }


from torch.utils.data import DataLoader

if __name__ == '__main__':
    """
    因为ROOT = train_dataset/ods/crop511
    该dataset.py, 要在../train_dataset, 即/home/sf/Documents/github_proj/carSiam　所在的项目根目录里运行
    否则dataset.py无法找到ROOT目录, 报错:
    error image: train_dataset/ods/crop511/mall_low_right_train_01/000789.00.x.jpg
    """
    my_cfg = '/home/sf/Documents/github_proj/carSiam/experiments/siamcar_r50/config.yaml'
    cfg.merge_from_file(my_cfg)
    train_dataset = TrkDataset()
    pick = train_dataset.pick[:10]
    dl_train = DataLoader(train_dataset,
                          batch_size=1,
                          num_workers=1,
                          pin_memory=True,
                          sampler=None)

    for i, data in enumerate(dl_train):
        if i > 0:
            break
        # template.shape= torch.Size([32, 3, 127, 127])
        # search.shape= torch.Size([32, 3, 255, 255])
        # label_cls=torch.Size([32, 5, 25, 25])
        # label_loc = torch.Size([32, 4, 5, 25, 25])
        # bbox = torch.Size([32, 4])

        print('idex={}, pick={} | template.shape= {} search.shape= {} | label_cls={} | \n bbox= {} '.format(
            i, pick[i],
            data['template'].shape,
            data['search'].shape,
            data['label_cls'].shape,
            data['bbox']))
    cv2.destroyAllWindows()
