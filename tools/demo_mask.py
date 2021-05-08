import os
import sys
sys.path.append('../')

import argparse
import cv2
import torch
from glob import glob
import numpy as np
import json

from carsot.core.config import cfg
from carsot.models.model_builder import ModelBuilder
from carsot.tracker.siamcar_mask_tracker import SiamCARMaskTracker
from carsot.utils.model_load import load_pretrain
from carsot.utils.misc import describe
from carsot.utils.log_helper import init_log
import logging
logger = logging.getLogger('global')

torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='SiamCAR demo')
parser.add_argument('--config', type=str, default='../experiments/siamcar_r50/config.yaml', help='config file')
parser.add_argument('--snapshot', type=str, default='./snapshot_r50/new_model.pth', help='model name')
parser.add_argument('--video_name', default='../test_dataset/Biker', type=str, help='videos or image files')
args = parser.parse_args()


def get_frames(video_name):
    if not video_name:
        cap = cv2.VideoCapture(0)

        # warmup
        for i in range(5):
            cap.read()
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(args.video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        print('run img')
        # images = sorted(glob(os.path.join(video_name, 'img', '*.jp*')))
        images = sorted(glob(os.path.join(video_name, '*.jp*')))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def set_model_no_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def main():
    # load config
    cfg.merge_from_file(args.config)
    init_log('global', logging.INFO)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).eval().to(device)
    set_model_no_grad(model)
    # logger.info("config \n{}".format(json.dumps(cfg, indent=4)))
    # logger.info("model\n{}".format(describe(model)))

    # build tracker
    tracker = SiamCARMaskTracker(model, cfg.TRACK)

    hp = {'lr': 0.3, 'penalty_k': 0.04, 'window_lr': 0.4}

    first_frame = True
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
        print(video_name)
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for idx, frame in enumerate(get_frames(args.video_name)):
        if first_frame:
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
            except:
                exit()
            # tracker.init(frame, init_rect)
            tracker.init_TemplateObj(frame, init_rect)
            first_frame = False
        else:
            print(idx, end='')
            outputs = tracker.track(frame, hp)
            if 'polygon' in outputs:
                # polygon = np.array(outputs['polygon']).astype(np.int32)
                # cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                #               True, (0, 255, 0), 3)
                # mask 是浮点数，mask=mask > THERSHOLD, 得到1,0矩阵
                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                mask = mask.astype(np.uint8)
                cv2.imshow('maskODS',mask)
                mask = np.stack([mask, mask*255, mask]).transpose(1, 2, 0)
                frame = cv2.addWeighted(frame, 0.77, mask, 0.23, -1)

                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              (0, 255, 255), 2)
            else:
                bbox = list(map(int, outputs['bbox']))
                cv2.rectangle(frame, (bbox[0], bbox[1]),
                              (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                              (0, 255, 0), 2)
            print('---')
            cv2.putText(frame, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(video_name, frame)
            cv2.waitKey(10)


if __name__ == '__main__':
    main()