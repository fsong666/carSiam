from carsot.utils.bbox import get_axis_aligned_bbox
import numpy as np
from glob import glob
import os
import cv2
from os import listdir
from os.path import join
from torch.nn import functional as F
import torch.nn as nn
import torch

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


def test_unfold():
    x = torch.arange(0, 1 * 3 * 15 * 15).float()
    x = x.view(1, 3, 15, 15)
    print(x)
    x1 = F.unfold(x, kernel_size=3, dilation=1, stride=1)
    print(x1.shape)
    B, C_kh_kw, L = x1.size()
    x1 = x1.permute(0, 2, 1)
    x1 = x1.view(B, L, -1, 3, 3)
    print(x1)


def test_mse():
    x = torch.tensor([0.6024, 0.5069, 0.7231, 0.6084, 0.8560, 0.7203, 0.8127, 0.6838, 0.6844, 0.5759])
    target= torch.tensor([0.7303, 0.7969, 0.7589, 0.7936, 0.7838, 0.7826, 0.7756, 0.7828, 0.7614, 0.7806])
    print('x=\n', x)
    print('traget=\n', target)
    print('loss=', F.mse_loss(x, target))

def test_BCE():
    x = torch.tensor([0.5928, 0.6887, 0.7932, 0.7884, 0.6843, 0.5887])
    target= torch.tensor([0.3448, 0.2981, 0.3293, 0.3975, 0.3972, 0.4643])
    print('x=\n', x)
    print('traget=\n', target)
    loss = nn.BCEWithLogitsLoss()
    print('bce_loss=', F.binary_cross_entropy_with_logits(x, target))
    print('loss=', loss(x, target))


def test(**dicts):
    for k in dicts:
        # print('k={}, v={}'.format(k, v))
        print('k={}'.format(k))

if __name__ == '__main__':
    # test_dir()
    # test_unfold()
    # test_BCE()
    dicts = {'loss':10, 'time': 0.2}
    test(**dicts)