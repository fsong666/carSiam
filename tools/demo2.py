import os
import sys
sys.path.append('../')

import argparse
import cv2
import torch
from glob import glob

from carsot.core.config import cfg
from carsot.models.model_builder import ModelBuilder
from carsot.tracker.siamcar_tracker import SiamCARTracker
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
        images = sorted(glob(os.path.join(video_name, 'img', '*.jp*')))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def main():
    cfg.merge_from_file(args.config)
    init_log('global', logging.INFO)
    # load config
    cfg.merge_from_file(args.config)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).eval().to(device)
    # logger.info("model\n{}".format(describe(model)))

    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    hp = {'lr': 0.3, 'penalty_k': 0.04, 'window_lr': 0.4}

    first_frame = False
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for idx, frame in enumerate(get_frames(args.video_name)):
        if not first_frame:
            cv2.putText(frame, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(video_name, frame)
            print('Press y button to start select ROI in this frame, Esc to quit this process'
                  '\nother button to next frame!')
            k = cv2.waitKey(0)
            if k == 27:
                return
            if k != ord('y'):
                continue

            init_rect = cv2.selectROI(video_name, frame, False, False)

            tracker.init(frame, init_rect)
            first_frame = True
        else:
            outputs = tracker.track(frame, hp)
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(frame, (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 255, 0), 2)
            cv2.putText(frame, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(video_name, frame)
            cv2.waitKey(50)


if __name__ == '__main__':
    main()