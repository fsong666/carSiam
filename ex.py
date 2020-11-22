from carsot.utils.bbox import get_axis_aligned_bbox
import numpy as np
from glob import glob
import os
import cv2
from os import listdir
from os.path import join

def test__tracker():
    gt_bbox = [214.38, 155.66, 245.78, 152.2, 253.62, 223.4, 222.22, 226.86]
    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
    print('{} {} {} {}'.format(cx, cy, w, h))
    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    print(gt_bbox_)

def run_test():
    video_name = '/home/sf/Documents/github_proj/carSiam/ods/cube_street'
    images = sorted(glob(os.path.join(video_name, '*.png')))
    img = cv2.imread(images[3])
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_dir():
    # dirs = './train_dataset'
    dirs = './ods'
    videos = sorted(listdir(dirs)) # 只得到目录里的名字, 不带路径
    print(videos)

    pys = sorted(glob(join(dirs, '*'))) # glob 得到绝对路劲
    print(pys)

if __name__ == '__main__':
    test_dir()