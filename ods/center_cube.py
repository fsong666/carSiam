from math import pi, sin, cos, tan, atan2, hypot, modf
import numpy as np
import time
import cv2
from ods.basePanorama import BasePanorama


def cot(angle):
    return 1 / tan(angle)


class ViewerCUbe(BasePanorama):
    def __init__(self, imgInShape, fov=None):
        super(ViewerCUbe, self).__init__(imgInShape[:2])
        self.edge = int(self.w / 4)  # 2pi/4==pi/2, correspond diameter
        self.outSize = (self.edge, self.edge, 3)
        self.rotation = None
        self.inv_rotation = None
        self.viewer_h = 0

    def get_rotation(self, viewer):
        (i, j) = viewer
        phi = i * 2 * pi / self.w
        rotation_z = np.array([[cos(phi), sin(phi), 0.],
                               [-sin(phi), cos(phi), 0.],
                               [0., 0., 1.]])

        theta = j * pi / self.h_ods
        latitude = pi * 0.5 - theta
        rotation_y = np.array([[cos(latitude), 0., sin(latitude)],
                               [0., 1., 0.],
                               [-sin(latitude), 0., cos(latitude)]])
        self.rotation = np.dot(rotation_y, rotation_z)
        return self.rotation

    def project_view(self, theta, phi):
        x = sin(theta) * cos(phi)
        y = sin(theta) * sin(phi)
        z = cos(theta)

        point = np.array([x, y, z]).reshape(-1, 1)

        assert self.rotation is not None
        (x, y, z) = np.dot(self.rotation, point).reshape(-1)

        phi = atan2(y, x)
        r = hypot(x, y)
        theta = atan2(r, z)
        return theta, phi

    def toCUbe(self, imgIn, viewer):
        self.get_rotation(viewer)
        imgOut = np.zeros(self.outSize, np.uint8)

        (u, v) = viewer
        u = int(u)
        v = int(v)
        half = int(self.edge * 0.5)
        rng_u = range(u - 2 * half, u + 2 * half)
        rng_v = range(v - half, v + half)

        print('toCUbe-----')
        for j in rng_v:
            for i in rng_u:
                phi = i * 2 * pi / self.w  # [0, 2pi]
                theta = j * pi / self.h  # [0, pi] z轴开角
                (theta, phi) = self.project_view(theta, phi)

                # get coordinates of image plane
                if theta < 0.615 or theta > 2.527:
                    (x, y, z) = (0, 0, 0)
                elif phi <= pi / 4 or phi > 7 * pi / 4:
                    x = 1
                    y = tan(phi)
                    z = cot(theta) / cos(phi)
                    z = z if -1 <= z <= 1 else 0
                else:
                    (x, y, z) = (0, 0, 0)

                # 半径为1，　直径为2的单位球体
                # 将有正负的值转化成图像坐标的绝对正值 所以(y+1) (1-z), /２, 然后比例放大edge倍
                (u, v) = (int(self.edge * (y + 1) / 2), int(self.edge * (1 - z) / 2))

                if u >= self.outSize[1]:
                    u = self.outSize[1] - 1
                elif u < 0:
                    u = 0
                if v >= self.outSize[0]:
                    v = self.outSize[0] - 1
                elif v < 0:
                    v = 0
                imgOut[v, u] = imgIn[j, i]
        return imgOut

    def reversToCube(self, imgIn, in_viewer):
        """
        324ms for cube without inv_rotation
        593ms for cube with inv_rotation
        """
        viewer = list(in_viewer).copy()
        viewer_h = viewer[1]
        if viewer[1] >= self.h_ods:
            viewer[1] -= self.h_ods
        cx = self.w // 2
        if viewer[0] >= cx:
            viewer[0] = viewer[0] - cx
        else:
            viewer[0] = viewer[0] + cx

        self.get_rotation(viewer)
        assert self.rotation is not None
        inv_rotation = np.linalg.inv(self.rotation)

        imgOut = np.zeros(self.outSize, np.uint8)
        # print('revers toCUbe-----')
        for j in range(self.edge):
            for i in range(self.edge):
                # 归化为edge为2的像素坐标
                # get normalized (a, b) from (i, j) in left cube
                a = 2.0 * float(i) / self.edge
                b = 2.0 * float(j) / self.edge
                # get 3d point (x, y, z) from (a, b) in left cube
                (x, y, z) = (1.0, a - 1.0, 1.0 - b)
                point = np.array([x, y, z]).reshape(-1, 1)

                (x, y, z) = np.dot(inv_rotation, point).reshape(-1)

                phi = atan2(y, x)  # [-pi, pi]
                r = hypot(x, y)
                theta = atan2(z, r)  # [-pi/2, pi/2] 维度，与x轴夹角
                uf = 2.0 * self.edge / pi * (phi + pi)  # [0, 2pi]
                vf = 2.0 * self.edge / pi * (pi / 2 - theta)  # [0, pi] z轴开角
                if viewer_h >= self.h_ods:
                    vf += self.h_ods

                if uf >= self.w:
                    uf = self.w - 1
                elif uf < 0:
                    uf = 0
                if viewer_h < self.h_ods <= vf:
                    vf = self.h_ods - 1
                elif viewer_h >= self.h_ods and vf >= self.h_ods * 2:
                    vf = self.h_ods * 2 - 1
                elif vf < 0:
                    vf = 0
                imgOut[j, i] = imgIn[int(vf), int(uf)]
        # cv2.imshow('cube', imgOut)
        return imgOut

    def show_cube(self, imgIn, viewer):
        cv2.imshow('viewer cube', self.reversToCube(imgIn, viewer))

    def toPlane(self, pano, viewer):
        return self.reversToCube(pano, viewer)

    def get_inv_rotation(self, in_viewer):
        """
        used once for each new predicted cube
        """
        viewer = list(in_viewer).copy()
        if viewer[1] >= self.h_ods:
            viewer[1] -= self.h_ods
        cx = self.w // 2
        if viewer[0] >= cx:
            viewer[0] = viewer[0] - cx
        else:
            viewer[0] = viewer[0] + cx

        self.get_rotation(viewer)
        assert self.rotation is not None
        self.inv_rotation = np.linalg.inv(self.rotation)
        self.viewer_h = in_viewer[1]
        return self.inv_rotation

    def location_panorama(self, pred_cube):
        """
        inv_rotation, viewer_h and cube
        used in many times
        :param pred_cube: predicted pt in cube
        :return: pt in ods
        """
        assert self.viewer_h > 0
        assert self.inv_rotation is not None
        (i, j) = pred_cube

        a = 2.0 * float(i) / self.edge
        b = 2.0 * float(j) / self.edge
        (x, y, z) = (1.0, a - 1.0, 1.0 - b)
        point = np.array([x, y, z]).reshape(-1, 1)

        (x, y, z) = np.dot(self.inv_rotation, point).reshape(-1)

        phi = atan2(y, x)  # [-pi, pi]
        r = hypot(x, y)
        theta = atan2(z, r)  # [-pi/2, pi/2] 维度，与x轴夹角
        uf = 2.0 * self.edge / pi * (phi + pi)  # [0, 2pi]
        vf = 2.0 * self.edge / pi * (pi / 2 - theta)  # [0, pi] z轴开角
        if self.viewer_h >= self.h_ods:
            vf += self.h_ods

        if uf >= self.w:
            uf = self.w - 1
        elif uf < 0:
            uf = 0
        if self.viewer_h < self.h_ods <= vf:
            vf = self.h_ods - 1
        elif self.viewer_h >= self.h_ods and vf >= self.h_ods * 2:
            vf = self.h_ods * 2 - 1
        elif vf < 0:
            vf = 0
        return int(uf), int(vf)

    def get_pointInPanorama(self, viewer, pt):
        self.get_inv_rotation(viewer)
        return self.location_panorama(pt)

    def show_bboxInPanorama(self, viewer, bbox, imgOut, depth, right=True):
        self.get_inv_rotation(viewer)
        (x, y, w, h) = bbox
        center_left = self.location_panorama((x + w / 2., y + h / 2.))

        depth_value = depth[center_left[1], center_left[0]][0]
        if 0 < depth_value < 100:
            baseline = 175.0
            disparity = baseline / depth_value
        else:
            disparity = 0

        thickness = 2
        # two column
        for i in [x, x + w - 1]:
            for j in range(y, y + h - 1):
                (u_left, v_left) = pt = self.location_panorama((i, j))
                cv2.circle(imgOut, (u_left, v_left), thickness, (255, 0, 0), thickness)
                if right:
                    u_right = round(u_left - disparity)
                    v_right = v_left + self.h_ods
                    cv2.circle(imgOut, (u_right, v_right), thickness, (0, 0, 255), thickness)

        # two rows
        for j in [y, y + h - 1]:
            for i in range(x + 1, x + w - 2):
                (u_left, v_left) = pt = self.location_panorama((i, j))
                cv2.circle(imgOut, (u_left, v_left), thickness, (255, 0, 0), thickness)
                if right:
                    u_right = round(u_left - disparity)
                    v_right = v_left + self.h_ods
                    cv2.circle(imgOut, (u_right, v_right), thickness, (0, 0, 255), thickness)

        center_right = [round(center_left[0] - disparity), round(center_left[1] + self.h_ods)]

        cv2.line(imgOut, (center_left[0], 0), (center_left[0], self.h - 1), (255, 0, 0), 1)
        cv2.line(imgOut, (center_right[0], 0), (center_right[0], self.h - 1), (0, 0, 255), 1)
        # print('{}:{} -- {} | {}'.format(depth_value, center_left, center_right, center_left[0] - center_right[0]))

    def show_maskInPanorama(self, viewer, mask, bbox, imgOut, depth, right=True, idx=None):
        """
        343ms
        135ms from if (mask[j, i] == np.array([255, 1, 255])).all()
        """
        self.get_inv_rotation(viewer)
        disparity = 0
        (x, y, w, h) = bbox
        if right:
            center_left = self.location_panorama((x + w / 2., y + h / 2.))

            depth_value = depth[center_left[1], center_left[0]][0]
            if 0 < depth_value < 100:
                baseline = 175.0
                disparity = baseline / depth_value
            else:
                disparity = 0

        new_mask = np.zeros(imgOut.shape, dtype=np.uint8)
        mask_h, mask_w, _ = mask.shape
        thickness = 2
        y_stop = min(y + h, mask_h)
        x_stop = min(x + w, mask_w)
        for j in range(y, y_stop):
            for i in range(x, x_stop):
                if (mask[j, i] == np.array([255, 1, 255])).all():
                    (u_left, v_left) = self.location_panorama((i, j))
                    # new_mask[v_left, u_left] = mask[j, i]
                    cv2.circle(new_mask, (u_left, v_left), thickness, (1, 1, 1), thickness)
                    if right:
                        u_right = round(u_left - disparity)
                        v_right = v_left + self.h_ods
                        # new_mask[v_right, u_right] = mask[j, i]
                        cv2.circle(new_mask, (u_right, v_right), thickness, (1, 1, 1), thickness)
        if idx is not None:
            # cv2.putText(new_mask, str(idx), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            name = './mask/{:04d}.png'.format(idx + 600)
            # cv2.imwrite(name, new_mask)
        # cv2.imshow('maskODS', new_mask)
        c = np.array([255, 1, 255], dtype=np.uint8)
        imgOut = cv2.addWeighted(imgOut, 0.8, new_mask*c, 0.2, -1)
        new_mask = (~new_mask.astype(np.bool)).astype(np.uint8)
        return imgOut, new_mask


def test_reversCorr():
    imgIn = './test_img/test.png'
    imgIn = cv2.imread(imgIn)
    cx = imgIn.shape[1] // 2
    # viewer = [cx + 300, 200]
    viewer = [450, 386]
    print('origin viewer:', viewer)

    cv2.circle(imgIn, tuple(viewer), 3, (0, 255, 0), 3)

    centerCube = ViewerCUbe(imgIn.shape)
    since = time.time()
    imgOut = centerCube.reversToCube(imgIn, viewer)  # 583ms
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}s {:.0f}ms'.format(
        modf(time_elapsed)[1], modf(time_elapsed)[0] * 1000))

    pt_cube = (100, 250)
    cv2.circle(imgOut, pt_cube, 3, (0, 255, 255), 3)
    start = time.time()
    pt_src = centerCube.get_pointInPanorama(viewer, pt_cube)
    print('time=', time.time() - start)
    # print_time(start)
    cv2.circle(imgIn, pt_src, 3, (0, 255, 255), 3)

    cv2.imshow('imgIn', imgIn)
    cv2.imshow('imgOut', imgOut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_cv():
    # imgIn = './test_img/test.png'
    imgIn = './test_img/street001.png'
    imgIn = cv2.imread(imgIn)
    cx = imgIn.shape[1] // 2
    # viewer = [cx + 300, 200]
    viewer = [450+300, 286+80]

    print('origin viewer:', viewer)

    cv2.circle(imgIn, tuple(viewer), 3, (0, 255, 0), 3)

    centerCube = ViewerCUbe(imgIn.shape)
    since = time.time()
    # imgOut = centerCube.toCUbe(imgIn, viewer)
    imgOut = centerCube.reversToCube(imgIn, viewer)  # 583ms
    print_time(since)

    cv2.imshow('imgIn', imgIn)
    cv2.imshow('imgOut', imgOut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('./test_img/street001_centerCube.png', imgOut)


def print_time(since):
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}s {:.0f}ms'.format(
        modf(time_elapsed)[1], modf(time_elapsed)[0] * 1000))


if __name__ == '__main__':
    test_cv()
    # test_reversCorr()
