import cv2
import numpy as np
from ods.center_cube import ViewerCUbe
from ods.panorama import Panorama
from collections import deque
import math
from os.path import join, isdir
import glob

outImgs = '/home/sf/Documents/github_proj/carSiam/demo/outImgs/'


# class BackGround(ViewerCUbe):
class BackGround(Panorama):
    def __init__(self, img, dq_length, fov=math.pi / 2.0):
        super(BackGround, self).__init__(img.shape, fov)
        self.mask = None
        self.total_mask = np.ones([self.h, self.w, 3], np.uint8)
        self.backGround = None
        self.reference = img.copy()  # init backGround reference
        self.reference_mask = np.ones([self.h, self.w, 3], np.uint8)
        self.iter = 3
        self.end_idx = 0
        self.bg = [[]] * self.iter   # saved result
        self.current_total_maskList = [[]] * self.iter   # saved result
        self.referFrameList = [img.copy()] * self.iter
        self.referMaskList = [np.ones([self.h, self.w, 3], np.uint8)] * self.iter
        self.dq_mask = deque(maxlen=dq_length)
        self.dq_frame = deque(maxlen=dq_length)
        self.dq_futureMask = deque(maxlen=dq_length)
        self.dq_futureFrame = deque(maxlen=dq_length)
        self.out_mask = None
        self.isOverlap = False
        self.bg_masks = None
        self.bg_frames = None

    def _set_mask(self, pt, thickness=3):
        self.mask[pt[1], pt[0]] = 0
        cv2.circle(self.mask, pt, thickness, (0, 0, 0), thickness)

    def get_maskBackGround2(self, viewer, bbox, depth, idx, right=True):
        """
        viewr must in center of object in ods
        边界处，左右视差，　右图里的bbox在右边界会有缝隙，mask不封闭
        目标从左进入边界与从右进入边界，由于左边界与右边界是同一条竖线即同时存在,所以运动方向不影响边界判断
        """
        self.get_inv_rotation(viewer)
        (x, y, w, h) = bbox
        center_left = self.location_panorama((x + w / 2., y + h / 2.))

        depth_value = depth[center_left[1], center_left[0]][0]
        if 30 < depth_value < 100:
            baseline = 175.0
            disparity = baseline / depth_value
        else:
            disparity = 0

        self.mask = np.ones([self.h, self.w], np.uint8)
        # two column
        for i in [x, x + w - 1]:
            for j in range(y, y + h - 1):
                (u_left, v_left) = pt = self.location_panorama((i, j))
                self._set_mask(pt)
                if right:
                    u_right = round(u_left - disparity)
                    v_right = v_left + self.h_ods
                    self._set_mask((u_right, v_right))

        # two rows
        # two flags are true at same time
        x_min_topRow = self.w
        x_min_bottomRow = self.w
        for j in [y, y + h - 1]:
            for i in range(x + 1, x + w - 2):
                (u_left, v_left) = pt = self.location_panorama((i, j))
                self._set_mask(pt)
                # j == y for topRow
                if j == y and u_left <= x_min_topRow:
                    x_min_topRow = u_left
                # j == y + h - 1 for bottomRow
                if j == y + h - 1 and u_left <= x_min_bottomRow:
                    x_min_bottomRow = u_left

                if right:
                    u_right = round(u_left - disparity)
                    v_right = v_left + self.h_ods
                    self._set_mask((u_right, v_right))
        h, w = self.mask.shape[:2]
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        # self.out_mask = (~self.total_mask.astype(np.bool)).astype(np.uint8).copy()
        # cv2.imshow('backGround out_mask', self.out_mask*255)
        mask = np.zeros([h + 2, w + 2], np.uint8)

        # viewr must in center of object in ods, otherwise foreground will be selected
        x, y = v = int(viewer[0]), int(viewer[1])
        self._fill_mask(mask, v, right=right)
        if min(x_min_topRow, x_min_bottomRow) == 0 and abs(x_min_topRow - x_min_bottomRow) <= 2:
            if x < self.w / 2:
                self._fill_mask(mask, (self.w - 1, y), right=right)
            else:
                self._fill_mask(mask, (0, y), right=right)
            print('x_min_topRow={}, x_min_bottomRow={}'.format(x_min_topRow, x_min_bottomRow))
            print('Top and bottom bbox’rows of {} frame are in boundary '.format(idx))
        self.total_mask *= self.mask

    def get_maskBackGround(self, viewer, bbox, depth, idx, right=True):
        (x, y, w, h) = bbox
        disparity = self._get_disparity(viewer, (int(x + w / 2.), int(y + h / 2.)), depth, min_value=30, right=right)
        x_map, y_map = self._get_mapXY(viewer)

        stop_y = max(0, min(y + h, self.planeHeight))
        stop_x = max(0, min(x + w, self.planeWidth))
        start_x = max(0, min(x, self.planeWidth - 1))
        start_y = max(0, min(y, self.planeHeight - 1))

        self.mask = np.ones([self.h, self.w], np.uint8)
        # two column
        for u in [start_x, stop_x - 1]:
            for v in range(start_y, stop_y):
                x, y = pt = int(x_map[v, u]), int(y_map[v, u])
                self._set_mask(pt)
                if right:
                    self._set_mask((round(x - disparity), y + self.h_ods))

        # two rows
        x_min_topRow = self.w
        x_min_bottomRow = self.w
        for v in [start_y, stop_y - 1]:
            for u in range(start_x, stop_x - 1):
                x, y = pt = int(x_map[v, u]), int(y_map[v, u])
                self._set_mask(pt)
                if v == start_y and x <= x_min_topRow:
                    x_min_topRow = x
                if v == stop_y - 1 and x <= x_min_bottomRow:
                    x_min_bottomRow = x

                if right:
                    self._set_mask((round(x - disparity), y + self.h_ods))

        h, w = self.mask.shape[:2]
        self.mask = cv2.cvtColor(self.mask, cv2.COLOR_GRAY2BGR)
        mask = np.zeros([h + 2, w + 2], np.uint8)

        # viewr must in center of object in ods, otherwise foreground will be selected
        x, y = v = int(viewer[0]), int(viewer[1])
        self._fill_mask(mask, v, right=right)
        if min(x_min_topRow, x_min_bottomRow) == 0 and abs(x_min_topRow - x_min_bottomRow) <= 1:
            if x < self.w / 2:
                self._fill_mask(mask, (self.w - 1, y), right=right)
            else:
                self._fill_mask(mask, (0, y), right=right)
            print('x_min_topRow={}, x_min_bottomRow={}'.format(x_min_topRow, x_min_bottomRow))
            print('Top and bottom bbox’rows of {} frame are in boundary '.format(idx))

        # object=0, background=1
        self.total_mask *= self.mask

    def _fill_mask(self, mask, viewer, right=True):
        cv2.floodFill(self.mask, mask, viewer, (0, 0, 0),
                      loDiff=0, upDiff=0, flags=cv2.FLOODFILL_FIXED_RANGE)
        if right:
            cv2.floodFill(self.mask, mask, (viewer[0], viewer[1] + self.h_ods), (0, 0, 0),
                          loDiff=0, upDiff=0, flags=cv2.FLOODFILL_FIXED_RANGE)

    def _select_referenceFrame(self, current_total_mask):
        assert len(self.dq_mask) > 0 and len(self.dq_frame) > 0
        if self.reference_mask is not None:
            if (self.reference_mask * current_total_mask).sum() == 0:
                return
        print('selecting', end="")
        for mask, frame in zip(self.dq_mask, self.dq_frame):
            # print(".", end='', flush=True)
            if (mask * current_total_mask).sum() == 0:
                self.reference = frame.copy()
                self.reference_mask = mask.copy()
                print('selected new backGround reference frame')
                return
        print('...end')

    def _update_referenceFrame2(self, current_total_mask, index=0):
        assert len(self.dq_mask) > 0 and len(self.dq_frame) > 0
        refer_overlap = (self.referMaskList[index] * current_total_mask).sum()
        if refer_overlap == 0:
            if index == 0:
                self.isOverlap = False
            return False
        print('updating...')
        min_overlap_idx = len(self.dq_mask)
        min_overlap = self.h_ods * self.w
        # 67ms for loop
        for idx, mask in enumerate(zip(self.dq_mask)):
            overlap = (mask * current_total_mask).sum()
            if overlap < min_overlap:
                min_overlap_idx = idx
                min_overlap = overlap
        if min_overlap_idx < len(self.dq_mask):
            if (self.dq_mask[min_overlap_idx] * current_total_mask).sum() < refer_overlap:
                self.referFrameList[index] = self.dq_frame[min_overlap_idx].copy()
                self.referMaskList[index] = self.dq_mask[min_overlap_idx].copy()
                print('selected new backGround reference frame')
        overlap = (self.referFrameList[0] * current_total_mask).sum() > 0
        if index == 0:
            self.isOverlap = overlap
        return overlap

    def __update_referenceFrame(self, current_total_mask, index=0, future=False):
        """
        :param current_total_mask:
        :param index:
        :param future:
        :return: overlap is true or not
        """
        if future:
            dq_mask = self.dq_futureMask
            dq_frame = self.dq_futureFrame
        else:
            dq_mask = self.dq_mask
            dq_frame = self.dq_frame
        assert len(dq_mask) > 0 and len(dq_frame) > 0
        refer_overlap = (self.referMaskList[index] * current_total_mask).sum()
        if refer_overlap == 0:
            if index == 0:
                self.isOverlap = False
            return False
        if future:
            print('updating future...')
        else:
            print('updating previous...')
        min_overlap_idx = len(dq_mask)
        min_overlap = self.h_ods * self.w
        # 67ms for loop
        # from dq find the reference frame with minimal overlap, dq_mask[min_overlap_idx]
        for idx, mask in enumerate(zip(dq_mask)):
            overlap = (mask * current_total_mask).sum()
            if overlap < min_overlap:
                min_overlap_idx = idx
                min_overlap = overlap
        if min_overlap_idx < len(dq_mask):
            if (dq_mask[min_overlap_idx] * current_total_mask).sum() < refer_overlap:
                self.referFrameList[index] = dq_frame[min_overlap_idx].copy()
                self.referMaskList[index] = dq_mask[min_overlap_idx].copy()
                print('selected new backGround reference frame')
        overlap = (self.referFrameList[0] * current_total_mask).sum() > 0
        if index == 0:
            self.isOverlap = overlap
        return overlap

    def _update_referenceFrame(self, current_total_mask, index=0, direction=1):
        """
        direction=0 as previous
        direction=1 as future
        direction=2 as previous and future directions
        """
        if direction == 0:
            return self.__update_referenceFrame(current_total_mask, index, future=False)
        elif direction == 1:
            return self.__update_referenceFrame(current_total_mask, index, future=True)
        else:
            print('search on previous and future directions')
            overlap = self.__update_referenceFrame(current_total_mask, index, future=False)
            if overlap:
                overlap = self.__update_referenceFrame(current_total_mask, index, future=True)
            return overlap

    def get_videoBackGround(self, img, update=False, direction=1):
        self.backGround = self.total_mask * img
        current_total_mask = (~self.total_mask.astype(np.bool)).astype(np.uint8)  # mask=1
        # cv2.imshow('current mask', current_total_mask.copy() * 255)
        # cv2.imwrite(outImgs + 'backGroundMaskODS.png', current_total_mask.copy() * 255)
        # cv2.imwrite(outImgs + 'backGround_origin.png', self.backGround)
        # cv2.imwrite(outImgs + 'frame.png', img)
        idx = 0

        if update:
            # search reference frames until foreground complete is eliminated or maximum iterations
            while idx < self.iter and self._update_referenceFrame(current_total_mask, index=idx,
                                                                  direction=direction):  # 3
                # get new current_total_mask1 except overlap region
                current_total_mask1 = current_total_mask * (~self.referMaskList[idx].astype(np.bool)).astype(np.uint8)
                # move pixels in region current_total_mask1 from reference frame to self.backGround
                self.backGround = current_total_mask1 * self.referFrameList[idx] + self.backGround
                self.current_total_maskList[idx] = current_total_mask1.copy()
                self.bg[idx] = self.backGround.copy()

                # run intersection, to get new current_total_mask with overlap for next iteration eliminate
                current_total_mask = current_total_mask * self.referMaskList[idx]
                idx += 1

            if 0 < idx < self.iter:
                print('early stop ', idx)
                self.end_idx = idx
                current_total_mask1 = current_total_mask * (~self.referMaskList[idx].astype(np.bool)).astype(np.uint8)
                self.bg[idx] = current_total_mask1 * self.referFrameList[idx] + self.backGround
                self.current_total_maskList[idx] = current_total_mask1.copy()
            else:
                if idx == self.iter:
                    self.end_idx = idx - 1

                    # self.backGround = current_total_mask * img + self.backGround

        else:
            if self.isOverlap:
                for i in range(self.end_idx + 1):
                    current_total_mask1 = current_total_mask * (~self.referMaskList[i].astype(np.bool)).astype(np.uint8)
                    self.backGround = current_total_mask1 * self.referFrameList[i] + self.backGround
                    self.current_total_maskList[i] = current_total_mask1.copy()
                    self.bg[i] = self.backGround.copy()

                    current_total_mask = current_total_mask * self.referMaskList[i]
            # self.backGround = current_total_mask * img + self.backGround

        # show backGround mask
        if self.isOverlap:
            for i in range(self.end_idx + 1):
                name_mask = 'backGround_mask{}'.format(i)
                name_referFrame = 'backGround_reference{}'.format(i)
                name_bg = 'backGround{}'.format(i)
                # cv2.imshow(name_mask, self.current_total_maskList[i] * 200)
                # cv2.imshow(name_referFrame, self.referFrameList[i])
                # cv2.imshow(name_bg, self.bg[i])

                # cv2.imwrite(outImgs + name_mask + '.png', self.current_total_maskList[i] * 255)
                # cv2.imwrite(outImgs + name_referFrame + '.png', self.referFrameList[i])
                # cv2.imwrite(outImgs + name_bg + '.png', self.bg[i])
        else:
            print('no overlap')
            self.backGround = current_total_mask * self.referFrameList[0] + self.backGround
            # cv2.imshow('backGround mask0', current_total_mask * 200)
            # cv2.imshow('backGround reference0', self.referFrameList[0])
        return self.backGround

    def get_videoBackGround2(self, img, update=False):
        self.backGround = self.total_mask * img
        current_total_mask = (~self.total_mask.astype(np.bool)).astype(np.uint8)
        if update:
            self._update_referenceFrame(current_total_mask.copy())
        if self.isOverlap:
            current_total_mask *= (~self.referMaskList[0].astype(np.bool)).astype(np.uint8)
            # self.backGround = (~current_total_mask.astype(np.bool)).astype(np.uint8) * img

        self.backGround = current_total_mask * self.referFrameList[0] + self.backGround
        # cv2.imshow('backGround reference', self.referFrameList[0])
        # cv2.imshow('backGround mask', current_total_mask * 200)
        return self.backGround

    def append_previousFrameAndBGMask(self, img, bg_mask=None):
        if bg_mask is not None:
            self.dq_mask.append(bg_mask.copy())
        else:
            current_total_mask = (~self.total_mask.astype(np.bool)).astype(np.uint8)
            self.dq_mask.append(current_total_mask.copy())

        self.dq_frame.append(img.copy())

    def reset(self):
        self.total_mask = np.ones([self.h, self.w, 3], np.uint8)

    def init_dq_future(self, bGroundMask_path, frame_path, start_idx=1, distance=50):
        assert isdir(bGroundMask_path) and isdir(frame_path)
        self.bg_masks = sorted(glob.glob(join(bGroundMask_path, '*.png')))
        self.bg_frames = sorted(glob.glob(join(frame_path, '*.png')))
        step = distance // self.dq_futureMask.maxlen
        for idx, (mask, frame) in enumerate(zip(self.bg_masks, self.bg_frames)):
            if idx > start_idx + distance:
                break
            if idx > start_idx and (idx - start_idx) % step == 0:
                mask = cv2.imread(mask)
                frame = cv2.imread(frame)
                cv2.putText(frame, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.dq_futureMask.append((mask // 255).copy())
                self.dq_futureFrame.append(frame.copy())
                # cv2.imshow('mask', mask)
                # cv2.imshow('frame', frame)
                # cv2.waitKey(500)

    def append_futureFrameAndBGMask(self, idx, distance=50):
        assert self.bg_frames is not None and self.bg_masks is not None
        if idx < len(self.bg_masks) - distance:
            mask = cv2.imread(self.bg_masks[idx + distance])
            frame = cv2.imread(self.bg_frames[idx + distance])
            cv2.putText(frame, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.dq_futureMask.append((mask // 255).copy())
            self.dq_futureFrame.append(frame.copy())

    def get_VideoBackGround_timeDirection(self, img=None, update=False, direction=1):
        """
        direction=0 as previous
        direction=1 as future
        direction=other as previous and future directions
        """
        if len(self.dq_mask) > 0:
            return self.get_videoBackGround(img, update, direction=direction)
        else:
            # previous mask is append from first sample
            # if frame is before first sample frame, only future mask are used
            return self.get_videoBackGround(img, update, direction=1)


if __name__ == '__main__':
    img = cv2.imread('../demo/mall/0601.png')
    img_path = '../demo/mall/'
    video = '../ods/dataset/Data/origin_data/mall_combined'
    bg = BackGround(img, 5)
    bg.init_dq_future(img_path, video, start_idx=100)
