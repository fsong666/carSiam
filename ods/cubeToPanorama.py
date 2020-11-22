import cv2
from math import pi, atan2, hypot
import numpy as np
from ods.panoramaToCube import outImgToXYZ


def convertToPanorama(imgIn):
    """
    6 cube maps fully to panorama convert
    """
    inSize = imgIn.shape  # (900, 1200) cube map
    imgOut = np.zeros((int(inSize[1] * 0.5), inSize[1], 3), np.uint8)
    outSize = imgOut.shape
    edge = inSize[1] / 4  # the length of each edge in pixels 300
    for i in range(inSize[1]):  # x 1200
        location = int(i / edge)
        if location == 1:
            rng = range(0, int(edge) * 3)  # 0-900
        else:
            rng = range(int(edge), int(edge) * 2)  # 300-900
        for j in rng:  # y 900
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

            # (r, g, b) = interpolation_cv(uf, vf, inSize, imgIn)
            uf = outSize[0] - 1 if uf >= outSize[1] else uf
            vf = outSize[1] - 1 if vf >= outSize[0] else vf
            imgOut[int(vf), int(uf)] = imgIn[j, i]
            # imgOut[j, i] = (int(round(r)), int(round(g)), int(round(b)))
    cv2.imshow('panoram', imgOut)
    return imgOut


def test_convertToPanorama():
    imgIn = cv2.imread('./test_img/1158mycube.png')
    inSize = imgIn.shape
    cx = inSize[1] // 2
    cy = inSize[0] // 2
    edge = 80

    bbox = [cx + edge, cy, 150, 50]
    cv2.rectangle(imgIn, (bbox[0], bbox[1]),
                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                  (0, 255, 0), 2)

    convertToPanorama(imgIn)

    cv2.imshow('input cube', imgIn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class ToPanorama(object):
    """
    according estimated bbox in cube draw bbox in panorama from input video,
    not real convert to panorama
    """
    def __init__(self, imgIn, imgOut):
        assert imgIn.shape[1] == imgOut.shape[1]  # w
        assert imgIn.shape[0] == int(imgOut.shape[1] * 3 / 4)  # h
        self.inSize = imgIn.shape

    def location_panoram(self, pt):
        assert isinstance(pt, tuple)
        (x, y) = pt
        edge = self.inSize[1] / 4
        location = int(x / edge)

        (x, y, z) = outImgToXYZ(x, y, location, edge)
        theta = atan2(y, x)  # range -pi to pi
        r = hypot(x, y)
        phi = atan2(z, r)  # range -pi/2 to pi/2
        uf = (2.0 * edge * (theta + pi) / pi)
        vf = (2.0 * edge * (pi / 2 - phi) / pi)
        uf = self.inSize[0] - 1 if uf >= self.inSize[1] else uf
        vf = self.inSize[1] - 1 if vf >= self.inSize[0] else vf

        return int(uf), int(vf)

    def show_bbox(self, bbox, imgOut):
        (x, y, w, h) = bbox

        # two height
        for i in [x, x+w-1]:
            for j in range(y, y+h-1):
                (u, v) = self.location_panoram((i, j))
                imgOut[v, u] = np.array([0, 255, 0], dtype=int)
        # two width
        for j in [y, y+h-1]:
            for i in range(x+1, x+w-2):
                (u, v) = self.location_panoram((i, j))
                imgOut[v, u] = np.array([0, 255, 0], dtype=int)

        # cx = x + w // 2
        # cy = y + h // 2
        # center = location_panoram(imgIn, (cx, cy), imgOut)
        # cv2.circle(imgOut, center, 3, (0, 255, 0), 3)
        cv2.imshow('imgOut', imgOut)


def test_classToPanorama():
    imgIn = cv2.imread('./test_img/1158mycube.png')
    imgOut = cv2.imread('./test_img/1158.png')
    inSize = imgIn.shape
    cx = inSize[1] // 2
    cy = inSize[0] // 2
    edge = 80

    bbox = [cx + edge, cy, 150, 50]
    cv2.rectangle(imgIn, (bbox[0], bbox[1]),
                  (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                  (0, 255, 0), 2)

    pano = ToPanorama(imgIn, imgOut)
    pano.show_bbox(bbox, imgOut)
    cv2.imshow('imgIn', imgIn)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_convertToPanorama()
    # test_classToPanorama()