import os
import sys
sys.path.append('../')

import numpy as np
import argparse
import cv2
import torch
from glob import glob
from os.path import join, isdir
import math
import time
from math import modf

from carsot.core.config import cfg
from carsot.models.model_builder import ModelBuilder
from carsot.tracker.siamcar_tracker import SiamCARTracker
from carsot.tracker.siamcar_mask_tracker import SiamCARMaskTracker
from carsot.utils.model_load import load_pretrain
from ods.center_cube import ViewerCUbe
from ods.panorama import Panorama
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
parser.add_argument('--reDetect', action='store_true', default=True, help='whether reDetection')
args = parser.parse_args()
outImgs = '/home/sf/Documents/github_proj/carSiam/demo/outImgs/missErfolg/'

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


def print_time(since, name=None):
    time_elapsed = time.time() - since
    print(name, ': {:.0f}s {:.0f}ms'.format(
        modf(time_elapsed)[1], modf(time_elapsed)[0] * 1000))


def main():
    if not os.path.exists(args.video_name):
        print('None args.video_name')
        return

    # load config
    cfg.merge_from_file(args.config)
    init_log('global', logging.INFO)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # hp = {'lr': 0.3, 'penalty_k': 0.04, 'window_lr': 0.4}
    hp = {'lr': 0.3, 'penalty_k': 0.0, 'window_lr': 0.0}

    panorama = None
    plane = None
    init_rect_plane = None
    plane_name = 'viewer plane.{}'
    depth = join(args.depth_img, args.video_name.split('.')[0], 'depth', 'left')
    assert isdir(depth)
    viewer_list = []
    model_list = []
    tracker_list = []
    update_list = []
    backGround = None
    bbox_list = []
    mask_list = []
    start_idx = 0
    distance = 50
    dq_length = 5
    step = distance // dq_length
    # fov = math.pi*2/3.0
    fov = math.pi/2.0
    total_mask = np.ones([1200, 1200, 3], np.uint8)

    first_frame = False
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for idx, (frame, depth) in enumerate(zip(get_frames(args.video_name), get_frames(depth))):
        if idx == 0:
            # panorama = ViewerCUbe(frame.shape)
            panorama = Panorama(frame.shape, fov=fov)
            backGround = BackGround(frame, dq_length, fov=fov)

        if not first_frame:
            cv2.putText(frame, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(video_name, frame)
            print('Press y button to start select ROI in plane map, Esc to quit this process'
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
                    plane = panorama.toPlane(frame, viewer)
                    cv2.imshow(plane_name.format(n), plane)
                    init_rect_plane = cv2.selectROI(plane_name.format(n), plane, False, False)

                except:
                    exit()
                model = load_model(device)
                model_list.append(model)
                tracker = SiamCARMaskTracker(model, cfg.TRACK)
                tracker.init(plane, init_rect_plane)  # init template und first center of bbox
                # tracker.init_TemplateObj(plane, init_rect_plane)
                tracker_list.append(tracker)
                bbox_list.append([])
                mask_list.append([])
                update_list.append([])
                print('Press q button to end, otherwise continue to start select ROI')
                n += 1

            assert len(tracker_list) > 0
            assert len(tracker_list) == len(viewer_list)
            print('tracker_list=', len(tracker_list))
            first_frame = True
            start_idx = idx + 1
        else:
            print(idx)
            sample = (idx - start_idx) % step == 0

            for i in range(len(tracker_list)):
                plane = panorama.toPlane(frame, viewer_list[i])
                planeCopy = plane.copy()
                cv2.putText(planeCopy, str(idx), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # cv2.imwrite(outImgs + 'planeIn.png', planeCopy)
                outputs = tracker_list[i].track(plane, hp, idx=idx, obj=i)
                bbox = list(map(int, outputs['bbox']))
                bbox_list[i] = bbox
                update_list[i] = outputs['update']
                color = (0, 255, 0) if i == 0 else (255, 0, 0)
                cv2.rectangle(plane,
                              (bbox[0], bbox[1]),
                              (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                              color, 2)
                if 'mask' in outputs:
                    mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                    mask = mask.astype(np.uint8)
                    mask = np.stack([mask, mask * 255, mask]).transpose(1, 2, 0)
                    mask_list[i] = mask
                    plane = cv2.addWeighted(plane, 0.8, mask, 0.2, -1)
                cv2.putText(plane, str(idx), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.imshow(plane_name.format(i), plane)
                # cv2.imwrite(outImgs + 'planeOut.png', plane)
            #     if sample or idx >= start_idx + distance:
            #         backGround.get_maskBackGround(viewer_list[i], bbox, depth, idx)
            #
            # if idx >= start_idx + distance:
            #     bg = backGround.get_videoBackGround(frame, update=sample)
            #     cv2.putText(bg, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            #     cv2.imshow('backGround', bg)
            cv2.putText(frame, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.imwrite(outImgs + 'frame.png', frame)
            # if sample:
            #     backGround.append_frame(frame)
            # backGround.reset()

            for i in range(len(tracker_list)):
                panorama.show_bboxInPanorama(viewer_list[i], bbox_list[i], frame, depth, obj=i)

                frame, bg_mask = panorama.show_maskInPanorama(viewer_list[i], mask_list[i], bbox_list[i], frame, depth, save=True)
                total_mask *= bg_mask
                bbox_center = getCenter(bbox_list[i])
                if update_list[i]:
                    # print('update viewer')
                    viewer_list[i] = panorama.get_pointInPanorama(viewer_list[i], bbox_center)
                # print('viewer=', viewer_list[i])
            current_total_mask = (~total_mask.astype(np.bool)).astype(np.uint8) * 255
            cv2.putText(current_total_mask, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            # cv2.imshow('bg_mask', current_total_mask)
            # name = './mall/{:04d}.png'.format(idx + 600)
            # if 90 <= idx <= 154:
                # cv2.imwrite(name, current_total_mask)
            total_mask = np.ones([1200, 1200, 3], np.uint8)

            cv2.imshow(video_name, frame)
            # trackingName = 'tracking{}.png'.format(idx)
            # cv2.imwrite(outImgs + trackingName, frame)
            # cv2.imshow('depth', depth)
            cv2.waitKey(1)
            print('----')


def getCenter(bbox):
    return [bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] / 2.]


if __name__ == '__main__':
    main()



