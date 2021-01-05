import cv2
import numpy as np
from ods.center_cube import ViewerCUbe


class BackGround(ViewerCUbe):
    def __init__(self, imgInShape):
        super(BackGround, self).__init__(imgInShape)
        self.mask = None
        self.total_mask = np.ones([self.h, self.w, 3], np.uint8)
        self.backGround = None

    def set_mask(self, pt, thickness=2):
        self.mask[pt[1], pt[0]] = 0
        cv2.circle(self.mask, pt, thickness, (0, 0, 0), thickness)

    def get_maskBackGround(self, viewer, bbox, depth, right=True):
        self.get_inv_rotation(viewer)
        (x, y, w, h) = bbox
        center_left = self.location_panorama((x + w / 2., y + h / 2.))

        depth_value = depth[center_left[1], center_left[0]][0]
        if 0 < depth_value < 100:
            baseline = 175.0
            disparity = baseline / depth_value
        else:
            disparity = 0

        self.mask = np.ones([self.h, self.w], np.uint8) * 255

        # two column
        for i in [x, x + w - 1]:
            for j in range(y, y + h - 1):
                (u_left, v_left) = pt = self.location_panorama((i, j))
                self.set_mask(pt)
                if right:
                    u_right = round(u_left - disparity)
                    v_right = v_left + self.h_ods
                    self.set_mask((u_right, v_right))

        # two rows
        for j in [y, y + h - 1]:
            for i in range(x + 1, x + w - 2):
                (u_left, v_left) = pt = self.location_panorama((i, j))
                self.set_mask(pt)
                if right:
                    u_right = round(u_left - disparity)
                    v_right = v_left + self.h_ods
                    self.set_mask((u_right, v_right))

        h, w = self.mask.shape[:2]
        mask = np.zeros([h + 2, w + 2], np.uint8)
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR) // 255

        fill(self.mask, mask, (int(viewer[0]), int(viewer[1])), (0, 0, 0), loDiff=0, upDiff=0)
        fill(self.mask, mask, (int(viewer[0]), int(viewer[1] + self.h_ods)), (0, 0, 0), loDiff=0, upDiff=0)
        self.total_mask *= self.mask

    def init_backGround(self, img):
        self.backGround = self.total_mask * img
        cv2.imshow('backGround', self.backGround)

    def get_backGround(self, img):
        self.total_mask = (~self.total_mask.astype(np.bool)).astype(np.uint8)
        self.backGround = self.total_mask * img + self.backGround
        cv2.imshow('backGround', self.backGround)
        return self.backGround


def fill(image, mask, seedPoint, newVal, loDiff=None, upDiff=None):
    loDiff = (loDiff, loDiff, loDiff)
    upDiff = (upDiff, upDiff, upDiff)
    cv2.floodFill(image, mask, seedPoint, newVal, loDiff=loDiff, upDiff=upDiff, flags=cv2.FLOODFILL_FIXED_RANGE)


