from math import pi, atan2, hypot, floor, modf
from numpy import clip
import time
import cv2
import numpy as np


def outImgToXYZ(i, j, location, edge):
    """
    get x,y,z coords from out image pixels coords
    :param i: pixel coord x
    :param j: pixel coord y
    :param location: location number in output image with 6 cubes
    :param edge: length of each cube face
    :return: coords of spherical coordinate with radial=1
    """
    a = 2.0 * float(i) / edge
    b = 2.0 * float(j) / edge
    if location == 0:  # back
        (x, y, z) = (-1.0 + a, -1.0, 3.0 - b)
    elif location == 1:  # left
        (x, y, z) = (1.0, a - 3.0, 3.0 - b)
    elif location == 2:  # front
        (x, y, z) = (-a + 5.0, 1.0, 3.0 - b)
    elif location == 3:  # right
        (x, y, z) = (-1.0, 7.0 - a, 3.0 - b)

    elif location == 4:  # top
        (x, y, z) = (b - 1.0, a - 3.0, 1.0)
    elif location == 5:  # bottom
        (x, y, z) = (3.0 - b, a - 3.0, -1.0)
    else:
        (x, y, z) = (0, 0, 0)
    return x, y, z


def interpolation(uf, vf, inSize, imgIn):
    """
    convert using an inverse transformation
    Use bilinear interpolation between the four surrounding pixels
    """
    ui = floor(uf)  # coord of pixel to bottom left
    vi = floor(vf)
    u2 = ui + 1  # coords of pixel to top right
    v2 = vi + 1
    mu = uf - ui  # fraction of way across pixel
    nu = vf - vi

    A = imgIn[int(clip(vi, 0, inSize[0] - 1)), ui % inSize[1]]
    B = imgIn[int(clip(vi, 0, inSize[0] - 1)), u2 % inSize[1]]
    C = imgIn[int(clip(v2, 0, inSize[0] - 1)), ui % inSize[1]]
    D = imgIn[int(clip(v2, 0, inSize[0] - 1)), u2 % inSize[1]]

    # interpolate
    (r, g, b) = (
        A[0] * (1 - mu) * (1 - nu) + B[0] * (mu) * (1 - nu) + C[0] * (1 - mu) * nu + D[0] * mu * nu,
        A[1] * (1 - mu) * (1 - nu) + B[1] * (mu) * (1 - nu) + C[1] * (1 - mu) * nu + D[1] * mu * nu,
        A[2] * (1 - mu) * (1 - nu) + B[2] * (mu) * (1 - nu) + C[2] * (1 - mu) * nu + D[2] * mu * nu)

    return r, g, b


def convertToCubes(imgIn):
    """
    Get cube image using inverse transformation
    from output cube to source equirectangular image.

    Rather than loop through each pixel in the source
    and find the corresponding pixel in the target,
    just loop through the target images
    and find the closest corresponding source pixel.

    Note:fest convert panorama to 6 cubes.
    157ms for one cube
    1s 7ms for 6 cubes
    """
    inSize = imgIn.shape  # (600, 1200)
    imgOut = np.zeros((int(inSize[1] * 3 / 4), inSize[1], 3), np.uint8)
    outSize = imgOut.shape
    edge = inSize[1] / 4  # the length of each edge in pixels 300
    for i in range(outSize[1]):  # # range of x-axis 1200
        location = int(i / edge)
        if location == 1:
            rng = range(0, int(edge * 3))  # range of y-axis 900
        else:
            rng = range(int(edge), int(edge) * 2)

        for j in rng:  # x
            location = int(i / edge)

            if j < edge:
                location2 = 4  # top
            elif j >= 2 * edge:
                location2 = 5  # bottom
            else:
                location2 = location

            (x, y, z) = outImgToXYZ(i, j, location2, edge)

            # Geographic coordinate system
            theta = atan2(y, x)  # range -pi to pi
            r = hypot(x, y)
            phi = atan2(z, r)  # range -pi/2 to pi/2
            uf = (2.0 * edge * (theta + pi) / pi)
            vf = (2.0 * edge * (pi / 2 - phi) / pi)

            # (r, g, b) = interpolation(uf, vf, inSize, imgIn)
            uf = inSize[0] - 1 if uf >= inSize[1] else uf
            vf = inSize[1] - 1 if vf >= inSize[0] else vf
            imgOut[j, i] = imgIn[int(vf), int(uf)]
            # imgOut[j, i] = (int(round(r)), int(round(g)), int(round(b)))

    return imgOut


class ToCUbe(object):
    """
    Convert panorama to the cube, where the estimated bbox appears.
    imgIn: panorama
    imgOut: cube map
    """

    def __init__(self, imgInshape):
        self.inSize = imgInshape
        self.edge = int(self.inSize[1] / 4)
        self.outSize = (int(self.inSize[1] * 3 / 4), self.inSize[1], 3)
        self.imgOut = None

    def _get_location(self, pt):
        (x, y) = pt
        assert 0 <= x < self.outSize[1]
        assert 0 <= y < self.outSize[0]

        location_x = int(x / self.edge)
        if y < self.edge:
            location = 4  # top
        elif y >= 2 * self.edge:
            location = 5  # bottom
        else:
            location = location_x
        return location

    def _toCube(self, imgIn, location):
        if location < 4:
            location_x = location
        else:
            location_x = 1

        if location == 4:
            location_y = 0
        elif location == 5:
            location_y = 2
        else:
            location_y = 1

        start_x = location_x * self.edge
        start_y = location_y * self.edge

        # print('start:', (start_x, start_y))
        for j in range(start_y, start_y + self.edge):
            for i in range(start_x, start_x + self.edge):
                (x, y, z) = outImgToXYZ(i, j, location, self.edge)
                theta = atan2(y, x)  # range -pi to pi
                r = hypot(x, y)
                phi = atan2(z, r)  # range -pi/2 to pi/2
                uf = (2.0 * self.edge * (theta + pi) / pi)
                vf = (2.0 * self.edge * (pi / 2 - phi) / pi)

                uf = self.inSize[0] - 1 if uf >= self.inSize[1] else uf
                vf = self.inSize[1] - 1 if vf >= self.inSize[0] else vf
                self.imgOut[j, i] = imgIn[int(vf), int(uf)]

    def show_cube(self, imgIn, bbox=None):
        """
        all cube to show if bbox is none
        """
        if bbox:
            (x, y, w, h) = bbox

            vertex1 = self._get_location((x, y))
            vertex2 = self._get_location((x + w - 1, y))
            vertex3 = self._get_location((x + w - 1, y + h - 1))
            vertex4 = self._get_location((x, y + h - 1))
            center = self._get_location((x + w // 2, y + h // 2))
            mid1 = self._get_location((x + w // 2, y))
            mid2 = self._get_location((x + w - 1, y + h // 2))
            mid3 = self._get_location((x + w // 2, y + h - 1))
            mid4 = self._get_location((x, y + h // 2))

            locations = [vertex1, vertex2, vertex3, vertex4, center,
                         mid1, mid2, mid3, mid4]
            locations = set(locations)
        else:  # all 6 face to show
            locations = list(range(6))

        self.imgOut = np.zeros(self.outSize, np.uint8)
        for face in locations:
            self._toCube(imgIn, face)

        return self.imgOut


def test_classToCUbe():
    imgIn = cv2.imread('./test_img/test.png')

    bbox = [220, 220, 450, 350]  # (x, y, w, h)
    to_cube = ToCUbe(imgIn.shape)
    since = time.time()
    imgOut = to_cube.show_cube(imgIn, bbox)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}s {:.0f}ms'.format(
        modf(time_elapsed)[1], modf(time_elapsed)[0] * 1000))  # 1s 152ms

    cv2.rectangle(imgOut, (bbox[0], bbox[1]),
                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                  (0, 255, 0), 2)
    cv2.imshow('imgIn', imgIn)
    cv2.imshow('imgOut', imgOut)

    k = cv2.waitKey(0)
    if k == ord('a'):
        print(k)
        cv2.destroyAllWindows()


def test_Tocube():
    # imgIn = cv2.imread('./test_img/test.png')
    imgIn = cv2.imread('./test_img/street001.png')
    since = time.time()
    imgOut = convertToCubes(imgIn)
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}s {:.0f}ms'.format(
        modf(time_elapsed)[1], modf(time_elapsed)[0] * 1000))  # 1s 57ms

    cv2.imshow('imgIn', imgIn)
    cv2.imshow('imgOut', imgOut)
    k = cv2.waitKey(0)
    if k == 27:
        cv2.destroyAllWindows()
    cv2.imwrite('streetCubeMap.png', imgOut)


if __name__ == '__main__':
    # test_classToCUbe()
    test_Tocube()

