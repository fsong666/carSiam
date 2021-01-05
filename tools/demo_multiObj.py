import os
import sys

sys.path.append('../')

import numpy as np
import argparse
import cv2
import torch
from glob import glob
from os.path import join, isdir
from os import listdir

from carsot.core.config import cfg
from carsot.models.model_builder import ModelBuilder
from carsot.tracker.siamcar_tracker import SiamCARTracker
from carsot.tracker.siamcar_mask_tracker import SiamCARMaskTracker
from carsot.utils.model_load import load_pretrain
from ods.center_cube import ViewerCUbe
from ods.background import BackGround
import logging
from carsot.utils.log_helper import init_log
logger = logging.getLogger('global')
torch.set_num_threads(1)

parser = argparse.ArgumentParser(description='SiamCAR demo')
parser.add_argument('--config', type=str, default='../experiments/siamcar_r50/config.yaml', help='config file')
parser.add_argument('--snapshot', type=str, default='./snapshot_r50/new_model.pth', help='model name')
parser.add_argument('--video_name', default='../test_dataset/Biker', type=str, help='videos or image files')
parser.add_argument('--depth_img', default='../ods/dataset/depth', type=str, help='depth images')
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
        # images = sorted(glob(os.path.join(video_name, 'img', '*.jp*')))
        images = sorted(glob(os.path.join(video_name, '*.png')))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def load_model(device=torch.device('cuda')):
    model = ModelBuilder()
    model = load_pretrain(model, args.snapshot).eval().to(device)
    for param in model.parameters():
        param.requires_grad = False
    return model


def main():
    if not os.path.exists(args.video_name):
        print('None args.video_name')
        return

    # load config
    cfg.merge_from_file(args.config)
    init_log('global', logging.INFO)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    hp = {'lr': 0.3, 'penalty_k': 0.04, 'window_lr': 0.4}

    toCube = None
    cube = None
    init_rect_cube = None
    cube_name = 'viewer cube.{}'
    mask_name = 'mask.{}'
    depth = join(args.depth_img, args.video_name.split('.')[0], 'depth', 'left')
    assert isdir(depth)
    viewer_list = []
    model_list = []
    tracker_list = []
    backGround = None
    bbox_list = []
    mask_list = []
    start_idx = 0

    first_frame = False
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for idx, (frame, depth) in enumerate(zip(get_frames(args.video_name), get_frames(depth))):
        if idx == 0:
            toCube = ViewerCUbe(frame.shape)
            backGround = BackGround(frame.shape)

        if not first_frame:
            cv2.putText(frame, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(video_name, frame)
            print('Press y button to start select ROI in cube map, Esc to quit this process'
                  '\nother button to next frame!')
            k = cv2.waitKey(0)
            if k == 27:
                return
            if k != ord('y'):
                continue

            n = 0
            while cv2.waitKey(0) != ord('q'):
                try:
                    init_rect = cv2.selectROI(video_name, frame, False, False)
                    print('init_rect_{}={}'.format(n, init_rect))  # init first viewer
                    viewer = [init_rect[0] + init_rect[2] / 2., init_rect[1] + init_rect[3] / 2.]
                    print('viewer_{}={}'.format(n, viewer))
                    assert viewer[1] < frame.shape[1]//2
                    viewer_list.append(viewer)
                    cube = toCube.reversToCube(frame, viewer)
                    cv2.imshow(cube_name.format(n), cube)
                    init_rect_cube = cv2.selectROI(cube_name.format(n), cube, False, False)
                except:
                    exit()
                model = load_model(device)
                model_list.append(model)
                # tracker = SiamCARTracker(model, cfg.TRACK)
                tracker = SiamCARMaskTracker(model, cfg.TRACK)
                tracker.init(cube, init_rect_cube)  # init template und first center of bbox
                tracker_list.append(tracker)
                bbox_list.append([])
                mask_list.append([])
                print('Press q button to end, otherwise continue to start select ROI')
                n += 1

            assert len(tracker_list) > 0
            assert len(tracker_list) == len(viewer_list)
            print('tracker_list=', len(tracker_list))
            first_frame = True
            start_idx = idx
        else:
            for i in range(len(tracker_list)):
                cube = toCube.reversToCube(frame, viewer_list[i])
                outputs = tracker_list[i].track(cube, hp)
                bbox = list(map(int, outputs['bbox']))
                bbox_list[i] = bbox
                cv2.rectangle(cube,
                              (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              (0, 255, 0), 2)
                if 'mask' in outputs:
                    mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                    mask = mask.astype(np.uint8)
                    mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                    mask_list[i] = mask
                    cube = cv2.addWeighted(cube, 0.8, mask, 0.2, -1)
                cv2.imshow(cube_name.format(i), cube)

                if idx == start_idx + 1:
                    backGround.get_maskBackGround(viewer_list[i], bbox, depth)
            if idx == start_idx + 1:
                backGround.init_backGround(frame)
            if idx == start_idx + 320:
                backGround.get_backGround(frame)

            for i in range(len(tracker_list)):
                toCube.show_bboxInODS(viewer_list[i], bbox_list[i], frame, depth)
                frame = toCube.show_maskInODS(viewer_list[i], mask_list[i], bbox_list[i], frame, depth)
                # update viewer, viewer=center of predicted bbox in current cube und frame
                viewer_list[i] = toCube.get_pointInPanorama(viewer_list[i], getCenter(bbox_list[i]))

            cv2.putText(frame, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(video_name, frame)
            # cv2.imshow('depth', depth)
            cv2.waitKey(10)


def getCenter(bbox):
    return [bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] / 2.]


if __name__ == '__main__':
    main()
