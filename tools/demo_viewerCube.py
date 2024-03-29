"""
直接在ods视频选出template跟踪，将ods预测出的bbox中心作为viewer
映射到cube
"""

import os
import sys
sys.path.append('../')

import argparse
import cv2
import torch
from glob import glob
from os.path import join
from os import listdir

from carsot.core.config import cfg
from carsot.models.model_builder import ModelBuilder
from carsot.tracker.siamcar_tracker import SiamCARTracker
from carsot.utils.model_load import load_pretrain
from ods.center_cube import ViewerCUbe
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
    cfg.CUDA = torch.cuda.is_available()
    device = torch.device('cuda' if cfg.CUDA else 'cpu')

    # create model
    model = ModelBuilder()

    # load model
    model = load_pretrain(model, args.snapshot).eval().to(device)

    # build tracker
    tracker = SiamCARTracker(model, cfg.TRACK)

    hp = {'lr': 0.3, 'penalty_k': 0.04, 'window_lr': 0.4}

    # panorama = None
    toCube = None
    init_rect = None
    cube_name = 'viewer cube'
    depth = join(args.depth_img, args.video_name.split('.')[0], 'depth', 'left')
    print(depth)

    first_frame = False
    if args.video_name:
        video_name = args.video_name.split('/')[-1].split('.')[0]
    else:
        video_name = 'webcam'
    cv2.namedWindow(video_name, cv2.WND_PROP_FULLSCREEN)
    for idx, (frame, depth) in enumerate(zip(get_frames(args.video_name), get_frames(depth))):
        if idx == 0:
            # toCube = ToCUbe(frame.shape)
            toCube = ViewerCUbe(frame.shape)

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
            try:
                init_rect = cv2.selectROI(video_name, frame, False, False)
                print(init_rect)
                viewer = [init_rect[0] + init_rect[2] / 2., init_rect[1] + init_rect[3] / 2.]
                print(viewer)
                # cube = toCube.reversToCube(frame, viewer)
                # cv2.imshow(cube_name, cube)
                # init_rect_cube = cv2.selectROI(cube_name, cube, False, False)
            except:
                exit()
            tracker.init(frame, init_rect)
            first_frame = True
        else:
            outputs = tracker.track(frame, hp)
            bbox = list(map(int, outputs['bbox']))
            viewer = [bbox[0] + bbox[2] / 2., bbox[1] + bbox[3] / 2.]
            viewer_left = [round(viewer[0]), round(viewer[1])]

            cv2.rectangle(frame,  # left
                          (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (255, 0, 0), 2)
            cv2.line(frame, (viewer_left[0], 0), (viewer_left[0], frame.shape[0] - 1), (255, 0, 0), 1)
            # cv2.rectangle(depth,
            #               (bbox[0], bbox[1]),
            #               (bbox[0]+bbox[2], bbox[1]+bbox[3]),
            #               (0, 255, 0), 2)

            depth_value = depth[viewer_left[1], viewer_left[0]][0]
            if 0 < depth_value < 100:
                baseline = 175.0
                cx_r = round(viewer[0] - baseline / depth_value)
                bbox[0] = cx_r - bbox[2] // 2
            bbox[1] += (frame.shape[0] // 2)
            viewer_right = [round(bbox[0] + bbox[2] / 2.), round(bbox[1] + bbox[3] / 2.)]

            cv2.rectangle(frame,  # right
                          (bbox[0], bbox[1]),
                          (bbox[0]+bbox[2], bbox[1]+bbox[3]),
                          (0, 0, 255), 2)
            cv2.line(frame, (viewer_right[0], 0), (viewer_right[0], frame.shape[0] - 1), (0, 0, 255), 1)
            print('{}:{} -- {} | {}'.format(depth_value, viewer_left, viewer_right, viewer_left[0] - viewer_right[0]))

            cv2.putText(frame, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imshow(video_name, frame)

            cube = toCube.reversToCube(frame, viewer)
            # cv2.imshow('depth', depth)
            cv2.imshow(cube_name, cube)
            cv2.waitKey(10)


if __name__ == '__main__':
    main()
