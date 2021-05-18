import numpy as np
import math
import cv2
import time
from math import modf
from ods.basePanorama import BasePanorama


class Panorama(BasePanorama):
    def __init__(self, imgInShape, fov=math.pi / 2.0):
        super(Panorama, self).__init__(imgInShape[:2], fov=fov)
        self.planeHeight = self.planeWidth = 300
        self.fov = fov
        assert fov < math.pi
        self.ankathete = self.planeWidth / (2.0 * math.tan(fov / 2.0))
        self.up = np.array([0.0, 1.0, 0.0])

    def _toXY(self, phi, theta):
        x = int(phi / (2 * math.pi) * self.w)
        y = int(theta / math.pi * self.h_ods)
        return x, y

    def _toAngle(self, pt):
        (x, y) = pt
        phi = x * 2 * math.pi / self.w
        theta = y * math.pi / self.h_ods
        return phi, theta

    def _get_mapXY(self, viewer):
        (phi, theta) = self._toAngle(viewer)   # 3dXYZ to polar coordinate
        # get unit vector from viewer to spherical of panorama
        direction_normalized = np.array([-math.cos(phi) * math.sin(theta), math.cos(theta),
                                         math.sin(phi) * math.sin(theta)])
        assert not np.array_equal(direction_normalized, self.up)
        plane_x_direction = np.cross(direction_normalized, self.up)
        plane_x_direction_normalized = plane_x_direction / np.sqrt(np.sum(plane_x_direction ** 2))
        plane_y_direction = np.cross(plane_x_direction, direction_normalized)
        plane_y_direction_normalized = plane_y_direction / np.sqrt(np.sum(plane_y_direction ** 2))

        # (shift_x, shift_y) mit center of plane als origin, positive y-axis is to up, x-axis to right
        shift_y = 0.5 * self.planeHeight - np.repeat(np.arange(self.planeHeight, dtype=np.float32).reshape(-1, 1),
                                                     self.planeWidth, axis=1)
        shift_x = -0.5 * self.planeWidth + np.repeat(np.arange(self.planeWidth, dtype=np.float32).reshape(1, -1),
                                                     self.planeHeight, axis=0)

        # get all coordinate R=(x_plane,y_plane,z_plane) in image plane
        x_plane = self.ankathete * direction_normalized[0] - shift_x * plane_x_direction_normalized[0] + shift_y * \
                  plane_y_direction_normalized[0]
        y_plane = self.ankathete * direction_normalized[1] - shift_x * plane_x_direction_normalized[1] + shift_y * \
                  plane_y_direction_normalized[1]
        z_plane = self.ankathete * direction_normalized[2] - shift_x * plane_x_direction_normalized[2] + shift_y * \
                  plane_y_direction_normalized[2]

        # convert 3dXYZ to polar coordinate
        length = np.sqrt(np.sum(np.stack((np.power(x_plane, 2), np.power(y_plane, 2), np.power(z_plane, 2))), axis=0))
        direction_theta = np.arccos(y_plane / length)            # [0, math.pi] from positive y-axis to r
        direction_phi = np.arctan2(-z_plane, x_plane) + math.pi  # [0, 2*math.pi]

        # convert 3d point in spherical panorama to point in ERP　of panorama
        x_map = direction_phi / (2 * math.pi) * self.w
        y_map = direction_theta / math.pi * self.h_ods

        return x_map, y_map

    def toPlane(self, pano, viewer):
        x_map, y_map = self._get_mapXY(viewer)
        plane = cv2.remap(pano, x_map, y_map, cv2.INTER_LINEAR)
        return plane

    def get_pointInPanorama(self, viewer, pt):
        (phi, theta) = self._toAngle(viewer)
        direction_normalized = np.array([-math.cos(phi) * math.sin(theta), math.cos(theta),
                              math.sin(phi) * math.sin(theta)])
        assert not np.array_equal(direction_normalized, self.up)
        plane_x_direction = np.cross(direction_normalized, self.up)
        plane_x_direction_normalized = plane_x_direction / np.sqrt(np.sum(plane_x_direction**2))
        plane_y_direction = np.cross(plane_x_direction, direction_normalized)
        plane_y_direction_normalized = plane_y_direction / np.sqrt(np.sum(plane_y_direction**2))

        shift_y = 0.5*self.planeHeight - np.repeat(np.arange(self.planeHeight, dtype=np.float32).reshape(-1, 1), self.planeWidth, axis=1)
        shift_x = -0.5*self.planeWidth + np.repeat(np.arange(self.planeWidth, dtype=np.float32).reshape(1, -1), self.planeHeight, axis=0)

        y = shift_y[int(pt[1]), int(pt[0])]
        x = shift_x[int(pt[1]), int(pt[0])]

        pt_plane = self.ankathete * direction_normalized - x * plane_x_direction_normalized + y * plane_y_direction_normalized

        length = np.sqrt(np.sum(pt_plane**2))
        direction_theta = np.arccos(pt_plane[1] / length)
        direction_phi = np.arctan2(-pt_plane[2], pt_plane[0]) + math.pi

        x_map = direction_phi / (2 * math.pi) * self.w
        y_map = direction_theta / math.pi * self.h_ods

        # return int(x_map), int(y_map)
        return x_map, y_map

    def toPanorama(self, plane, viewer, bbox_range=None, disparity=0, right=False, save=False):
        (phi, theta) = self._toAngle(viewer)
        direction_normalized = np.array([-math.cos(phi) * math.sin(theta), math.cos(theta),
                              math.sin(phi) * math.sin(theta)])
        assert not np.array_equal(direction_normalized, self.up)
        plane_x_direction = np.cross(direction_normalized, self.up)
        plane_x_direction_normalized = plane_x_direction / np.sqrt(np.sum(plane_x_direction**2))
        plane_y_direction = np.cross(plane_x_direction, direction_normalized)
        plane_y_direction_normalized = plane_y_direction / np.sqrt(np.sum(plane_y_direction**2))

        shift_y = 0.5*self.planeHeight - np.repeat(np.arange(self.planeHeight, dtype=np.float32).reshape(-1, 1), self.planeWidth, axis=1)
        shift_x = -0.5*self.planeWidth + np.repeat(np.arange(self.planeWidth, dtype=np.float32).reshape(1, -1), self.planeHeight, axis=0)

        start_y = start_x = 0
        stop_x = stop_y = plane.shape[0]
        if bbox_range:
            (start_x, start_y, stop_x, stop_y) = bbox_range
            shift_y = shift_y[start_y:stop_y, start_x:stop_x]
            shift_x = shift_x[start_y:stop_y, start_x:stop_x]

        x_plane = self.ankathete * direction_normalized[0] - shift_x * plane_x_direction_normalized[0] + shift_y * plane_y_direction_normalized[0]
        y_plane = self.ankathete * direction_normalized[1] - shift_x * plane_x_direction_normalized[1] + shift_y * plane_y_direction_normalized[1]
        z_plane = self.ankathete * direction_normalized[2] - shift_x * plane_x_direction_normalized[2] + shift_y * plane_y_direction_normalized[2]

        length = np.sqrt(np.sum(np.stack((np.power(x_plane, 2), np.power(y_plane, 2), np.power(z_plane,2))), axis=0))
        direction_theta = np.arccos(y_plane / length)
        direction_phi = np.arctan2(-z_plane, x_plane) + math.pi

        x_map = direction_phi / (2 * math.pi) * self.w
        y_map = direction_theta / math.pi * self.h_ods

        imgOut = np.zeros((self.h, self.w, 3)).astype(np.uint8)
        thickness = 2
        for v in range(start_y, stop_y):
            for u in range(start_x, stop_x):
                vv = v - start_y
                uu = u - start_x
                y, x = int(y_map[vv, uu]), int(x_map[vv, uu])
                # y, x = int(y_map[v, u]), int(x_map[v, u])
                if x >= self.w:
                    x = self.w - 1
                if save:
                    if (plane[v, u] == np.array([255, 1, 255])).all():
                        cv2.circle(imgOut, (x, y), thickness, (1, 1, 1), thickness)
                        if right:
                            cv2.circle(imgOut, (x - disparity, y + self.h_ods), thickness, (1, 1, 1), thickness)
                else:
                    imgOut[y, x] = plane[v, u]
                    if right:
                        imgOut[y + self.h_ods, x - disparity] = plane[v, u]

        return imgOut

    def _get_disparity(self, viewer, center, depth, min_value=0, right=False):
        disparity = 0
        if right:
            # center_left = self.get_pointInPanorama(viewer, center)
            # cx, cy = round(center_left[1]), round(center_left[0])
            # depth_value = depth[cx, cy][0]
            # depth_value = depth[cx-1:cx+1, cy-1:cy+1].mean()
            depth_value = self.getDepthInPanorama(viewer, center, depth)
            if min_value < depth_value < 100:
                baseline = 175.0
                disparity = int(baseline / depth_value)
            else:
                disparity = 0
        return disparity

    def getDepthInPanorama(self, viewer, bbox_center, depthImg, r=1):
        center_left = self.get_pointInPanorama(viewer, bbox_center)
        cx, cy = round(center_left[1]), round(center_left[0])
        depth_value = depthImg[cx - r:cx + r, cy - r:cy + r].mean()
        return depth_value

    def show_maskInPanorama(self, viewer, mask, bbox, imgOut, depth, right=True, save=False):
        """
        81ms faster than 328ms for   right man in street
        """
        (x, y, w, h) = bbox
        disparity = self._get_disparity(viewer, (int(x + w / 2.), int(y + h / 2.)), depth, right=right)

        mask_h, mask_w, _ = mask.shape
        stop_y = max(0, min(y + h, mask_h))
        stop_x = max(0, min(x + w, mask_w))
        start_x = max(0, min(x, mask_w-1))
        start_y = max(0, min(y, mask_h-1))

        mask_inODS = self.toPanorama(mask, viewer, bbox_range=(start_x, start_y, stop_x, stop_y), disparity=disparity, right=right, save=save)

        if save:
            c = np.array([255, 1, 255], dtype=np.uint8)
            imgOut = cv2.addWeighted(imgOut, 0.8, mask_inODS*c, 0.2, -1)
            # imgOut = cv2.addWeighted(imgOut, 0.9, mask_inODS*c, 0.1, -1)
            mask_inODS = (~mask_inODS.astype(np.bool)).astype(np.uint8)
            # cv2.imshow('mask_ODS', mask_inODS*c)
            return imgOut, mask_inODS
        else:
            imgOut = cv2.addWeighted(imgOut, 0.95, mask_inODS, 0.05, -1)
            return imgOut

    def show_bboxInPanorama(self, viewer, bbox, imgOut, depth, right=True, obj=0):
        (x, y, w, h) = bbox
        disparity = self._get_disparity(viewer, (int(x + w / 2.), int(y + h / 2.)), depth, right=right)
        x_map, y_map = self._get_mapXY(viewer)

        stop_y = max(0, min(y + h, self.planeHeight))
        stop_x = max(0, min(x + w, self.planeWidth))
        start_x = max(0, min(x, self.planeWidth - 1))
        start_y = max(0, min(y, self.planeHeight - 1))
        thickness = 2
        color = (0, 255, 0) if obj == 0 else (255, 0, 0)
        # two column
        for u in [start_x, stop_x-1]:
            for v in range(start_y, stop_y):
                y, x = int(y_map[v, u]), int(x_map[v, u])
                cv2.circle(imgOut, (x, y), thickness, color, thickness)
                if right:
                    cv2.circle(imgOut, (x - disparity, y + self.h_ods), thickness, color, thickness)

        # two rows
        for v in [start_y, stop_y-1]:
            for u in range(start_x, stop_x):
                y, x = int(y_map[v, u]), int(x_map[v, u])
                cv2.circle(imgOut, (x, y), thickness, color, thickness)
                if right:
                    cv2.circle(imgOut, (x - disparity, y + self.h_ods), thickness, color, thickness)

        return imgOut


def print_time(since, name=None):
    time_elapsed = time.time() - since
    print(name, ': {:.0f}s {:.0f}ms'.format(
        modf(time_elapsed)[1], modf(time_elapsed)[0] * 1000))


# base = './test_img/'
base = './odsTransforTest/'


def test(imgName=None, idx=0, viewer=(600, 300)):
    img = base + imgName

    img = cv2.imread(img)
    h, w = img.shape[:2]
    x, y = viewer
    assert 0 <= x < w
    assert 0 < y < h

    cv2.putText(img, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.circle(img, viewer, 3, (0, 255, 0), 3)
    cv2.imshow('input', img)

    # pano = Panorama(img.shape, fov=math.pi*7/12.0)
    pano = Panorama(img.shape, fov=math.pi/2.0)

    since = time.time()
    plane = pano.toPlane(img, viewer)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}s {:.0f}ms'.format(
        modf(time_elapsed)[1], modf(time_elapsed)[0] * 1000))

    cv2.putText(plane, str(idx), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('plane', plane)

    # name = './test_img/middle_{}.jpg'.format('pi712')
    # cv2.imwrite("./test_img/street001_rayTracing.png", plane)

    figName = imgName.split('.')[0]
    viewer_p = '_'+ str(viewer[0]) + '_' + str(viewer[1])

    name = base + 'plane_' + figName + viewer_p + '.png'
    cv2.imwrite(name, plane)

    cv2.waitKey(0)


def test_getPt():
    plane = './test_img/middle.jpg'
    panoImg = './test_img/0600left.png'
    plane = cv2.imread(plane)
    panoImg = cv2.imread(panoImg)
    viewer = (600, 300)
    pt = (50, 200)
    cv2.circle(plane, pt, 3, (0, 255, 255), 3)
    cv2.imshow('plane', plane)
    pano = Panorama(panoImg.shape)

    since = time.time()
    out_pt = pano.get_pointInPanorama(viewer, pt)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}s {:.0f}ms'.format(
        modf(time_elapsed)[1], modf(time_elapsed)[0] * 1000))

    cv2.circle(panoImg, out_pt, 3, (0, 255, 255), 3)
    cv2.imshow('outPano', panoImg)
    cv2.waitKey(0)


def test_toPano():
    # plane = './test_img/middle.jpg'
    plane = './test_img/street001_rayTracing.png'
    panoImg = './test_img/0600left.png'
    plane = cv2.imread(plane)
    panoImg = cv2.imread(panoImg)
    viewer = (440, 400)
    cv2.imshow('plane', plane)
    pano = Panorama(panoImg.shape)

    since = time.time()
    outPano = pano.toPanorama(plane, viewer)
    print_time(since, 'toPano')

    cv2.imshow('outPano', outPano)
    # cv2.imwrite("./test_img/street001_reverseRayTracing.png", outPano)
    cv2.waitKey(0)


def test_ods(imgName, viewer):
    idx = int(imgName.split('_')[1].split('.')[0])
    return test(imgName, idx, viewer)


def test_odsToPano(planeImgName, directory):
    string = planeImgName.split('_')
    panoName = string[1] + '_' + string[2] + '.png'
    viewer = (int(string[3]), int(string[-1].split('.')[0]))
    print(panoName)
    print(viewer)
    panoImg = base + panoName
    plane = base + directory + '/' + planeImgName

    plane = cv2.imread(plane)
    panoImg = cv2.imread(panoImg)
    cv2.imshow('plane', plane)

    pano = Panorama(panoImg.shape)

    since = time.time()
    outPano = pano.toPanorama(plane, viewer)
    print_time(since, 'toPano')

    idx = string[2]
    cv2.putText(outPano, idx, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('outPano', outPano)
    name = base + 'pano' + planeImgName[5:]
    cv2.imwrite(name, outPano)
    cv2.waitKey(0)

if __name__ == '__main__':
    # test_ods(imgName='caveL_600.png', viewer=(0, 350))
    # test_toPano()
    # test_odsToPano('plane_streetL_0_1199_300.png', 'differentScene')
    test_odsToPano('plane_streetL_0_1199_300.png', 'street')