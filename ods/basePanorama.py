import math


class BasePanorama(object):
    def __init__(self, imgInShape, fov=math.pi / 2.0):
        (self.h, self.w) = imgInShape[:2]  # 600, 1200   1200, 1200
        self.h_ods = self.w // 2
        self.fov = fov

    def toPlane(self, pano, viewer):
        pass

    def toPanorama(self, plane, viewer, bbox_range=None, disparity=0, right=False):
        pass

    def get_pointInPanorama(self, viewer, pt):
        pass

    def show_maskInPanorama(self, viewer, mask, bbox, imgOut, depth, right=True):
        pass

    def show_bboxInPanorama(self, viewer, bbox, imgOut, depth, right=True):
        pass

