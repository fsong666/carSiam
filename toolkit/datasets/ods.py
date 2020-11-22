import os
import cv2
import json
import numpy as np

from glob import glob
from tqdm import tqdm
from PIL import Image

from .dataset import Dataset
from .video import Video


class ODSVideo(Video):
    """
    Args:
        name: video name
        root: dataset root
        video_dir: video directory
        init_rect: init rectangle
        img_names: image names
        gt_rect: groundtruth rectangle
    """

    def __init__(self, name, root, video_dir, init_rect, img_names, gt_rect,
                 load_img=False):
        super(ODSVideo, self).__init__(name, root, video_dir,
                                       init_rect, img_names, gt_rect, None, load_img)

        if not load_img:
            img_name = os.path.join(root, self.img_names[0])
            img = np.array(Image.open(img_name), np.uint8)
            self.width = img.shape[1]
            self.height = img.shape[0]


class ODSDataset(Dataset):
    """
    Args:
        name: dataset name, should be 'ODS'
        dataset_root: dataset root
        load_img: wether to load all imgs
    """

    def __init__(self, name, dataset_root, load_img=False):
        super(ODSDataset, self).__init__(name, dataset_root)
        with open(os.path.join(dataset_root, name + '.json'), 'r') as f:
            meta_data = json.load(f)

        # load videos
        pbar = tqdm(meta_data.keys(), desc='loading ' + name, ncols=100)
        self.videos = {}
        for video in pbar:
            pbar.set_postfix_str(video)
            self.videos[video] = ODSVideo(video,
                                          dataset_root,
                                          meta_data[video]['video_dir'],
                                          meta_data[video]['init_rect'],
                                          meta_data[video]['img_names'],
                                          meta_data[video]['gt_rect'])



