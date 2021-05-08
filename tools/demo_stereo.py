import os
import sys

sys.path.append('../')

import argparse
import cv2
import torch
import math
from glob import glob
from os.path import join, isdir
from os import listdir

from carsot.core.config import cfg
from carsot.models.model_builder import ModelBuilder
from carsot.tracker.siamcar_tracker import SiamCARTracker
from carsot.utils.model_load import load_pretrain
from ods.center_cube import ViewerCUbe
from ods.panorama import Panorama
from carsot.utils.log_helper import init_log
import logging
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


def main():
    if not os.path.exists(args.video_name):
        print('None args.video_name')
        return

    # load config
    cfg.merge_from_file(args.config)
    init_log('global', logging.INFO)
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).eval().to(device)

    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    hp = {'lr': 0.3, 'penalty_k': 0.04, 'window_lr': 0.4}

    panorama = None
    plane = None
    init_rect_plane = None
    viewer = None
    plane_name = 'viewer plane'
    depth = join(args.depth_img, args.video_name.split('.')[0], 'depth', 'left')
    assert isdir(depth)

    first_frame = False
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for idx, (frame, depth) in enumerate(zip(get_frames(args.video_name), get_frames(depth))):
        if idx == 0:
            # toplane = ViewerCUbe(frame.shape)
            panorama = Panorama(frame.shape, fov=math.pi/2.0)
            # panorama = Panorama(frame.shape, fov=2*math.pi/3.0)

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
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
                print('init_rect_{}={}'.format(0, init_rect))  # init first viewer
                viewer = [init_rect[0] + init_rect[2] / 2., init_rect[1] + init_rect[3] / 2.]
                print('viewer_{}={}'.format(0, viewer))
                assert viewer[1] < frame.shape[1] // 2
                plane = panorama.toPlane(frame, viewer)
                cv2.imshow(plane_name, plane)
                init_rect_plane = cv2.selectROI(plane_name, plane, False, False)
            except:
                exit()
            tracker.init(plane, init_rect_plane)  # init template und first center of bbox
            first_frame = True
        else:
            plane = panorama.toPlane(frame, viewer)
            outputs = tracker.track(plane, hp)
            bbox = list(map(int, outputs['bbox']))
            cv2.rectangle(plane,
                          (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                          (0, 255, 0), 2)
            predict_ptInplane = [bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] / 2.]

            panorama.show_bboxInPanorama(viewer, bbox, frame, depth)
            cv2.putText(frame, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(video_name, frame)
            # cv2.imshow('depth', depth)
            cv2.imshow(plane_name, plane)
            cv2.waitKey(10)

            # update viewer, viewer=center of predicted bbox in current plane und frame
            viewer = panorama.get_pointInPanorama(viewer, predict_ptInplane)


if __name__ == '__main__':
    main()
