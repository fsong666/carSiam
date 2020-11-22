from math import pi, sin, cos, tan, atan2, hypot, modf
import numpy as np
import time
import cv2


def cot(angle):
    return 1 / tan(angle)


class ViewerCUbe(object):
    def __init__(self, imgInShape):
        (self.h, self.w) = imgInShape[:2]  # 600, 1200   1200, 1200
        self.h_ods = self.w // 2
        # print('init ViewerCUbe ', (self.h, self.w))
        self.edge = int(self.w / 4)  # 2pi/4==pi/2
        self.outSize = (self.edge, self.edge, 3)
        self.rotation = None

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

    def reversToCube(self, imgIn, viewer):
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
                a = 2.0 * float(i) / self.edge
                b = 2.0 * float(j) / self.edge
                (x, y, z) = (1.0, a - 1.0, 1.0 - b)
                point = np.array([x, y, z]).reshape(-1, 1)

                (x, y, z) = np.dot(inv_rotation, point).reshape(-1)

                phi = atan2(y, x)  # [-pi, pi]
                r = hypot(x, y)
                theta = atan2(z, r)  # [-pi/2, pi/2] 维度，与x轴夹角
                uf = 2.0 * self.edge / pi * (phi + pi)        # [0, 2pi]
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


def test_cv():
    imgIn = './test_img/test.png'
    imgIn = cv2.imread(imgIn)
    cx = imgIn.shape[1]//2
    viewer = [cx + 300, 200]
    print('origin viewer:', viewer)

    cv2.circle(imgIn, tuple(viewer), 3, (0, 255, 0), 3)

    centerCube = ViewerCUbe(imgIn.shape)
    since = time.time()
    # imgOut = centerCube.toCUbe(imgIn, viewer)
    imgOut = centerCube.reversToCube(imgIn, viewer) # 583ms
    # centerCube.show_cube(imgIn, viewer)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}s {:.0f}ms'.format(
        modf(time_elapsed)[1], modf(time_elapsed)[0] * 1000))

    cv2.imshow('imgIn', imgIn)
    # cv2.imshow('imgOut', imgOut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_cv()