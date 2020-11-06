from pysot.utils.bbox import get_axis_aligned_bbox
import numpy as np


def test__tracker():
    gt_bbox = [214.38, 155.66, 245.78, 152.2, 253.62, 223.4, 222.22, 226.86]
    cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
    print('{} {} {} {}'.format(cx, cy, w, h))
    gt_bbox_ = [cx - (w - 1) / 2, cy - (h - 1) / 2, w, h]
    print(gt_bbox_)


if __name__ == '__main__':
    test__tracker()