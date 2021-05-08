import numpy as np
import cv2
import math
import time
from math import modf

import argparse


def myRemap(src, map_x, map_y):
    h, w = map_x_fast.shape[:2]
    imgOut = np.zeros((h, w, 3)).astype(np.uint8)
    for v in range(h):
        for u in range(w):
            imgOut[v, u] = src[int(map_y[v, u]), int(map_x[v, u])]
    cv2.imshow('my imgOut', imgOut)


# maps part of a panorama image onto a plane image
# example usage: python mapPanoToPlane 0600.png output.png
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('panoramapath', help='path a panorama')
    # parser.add_argument('outputpath', default='output.png', help='path to output image')
    parser.add_argument('--height', type=int, default=250, help='height of output view')
    parser.add_argument('--width', type=int, default=250, help='width of output view')
    parser.add_argument('--fov', type=float, default=(math.pi/2.0), help='field of view of the output')
    parser.add_argument('--phi', type=float, default=(4*math.pi/3), help='middlepoint phi of the view in the pano') # in [0, 2*math.pi]
    parser.add_argument('--theta', type=float, default=(math.pi/3.0), help='middlepoint theta of the view in the panoFN2') # in [0, math.pi]
    args = parser.parse_args()

    # read the panorama and its shape
    panorama = cv2.imread(args.panoramapath)
    panoheight, panowidth, channel = panorama.shape
    # x = int(args.phi / (2*math.pi) * panowidth)
    # y = int(args.theta / math.pi * panoheight)
    # cv2.circle(panorama, (x, y), 3, (0, 255, 0), 3)
    # cv2.imshow('imgIn', panorama)
    since = time.time()

    # initialized the output image along with the matrices needed for cv2.remap (they need to be of type np.float32!)
    output = np.zeros((args.height, args.width, channel))
    map_x = np.zeros((args.height, args.width), dtype=np.float32)
    map_y = np.zeros((args.height, args.width), dtype=np.float32)

    # compute the (x, y, z) direction vector from the spherical direction where we are looking at (theta, phi)
    # is normalized vector
    direction = np.array([-math.cos(args.phi) * math.sin(args.theta), math.cos(args.theta), math.sin(args.phi) * math.sin(args.theta)])
    up = np.array([0.0, 1.0, 0.0])
    assert not np.array_equal(direction, up)
    
    # compute the directions vectors for the image plane in x and y axis per pixel
    plane_x_direction = np.cross(direction, up)  # plane of d x up
    plane_x_direction_normalized = plane_x_direction / np.sqrt(np.sum(plane_x_direction**2))
    plane_y_direction = np.cross(plane_x_direction, direction)
    plane_y_direction_normalized = plane_y_direction / np.sqrt(np.sum(plane_y_direction**2))

    # anything else doesnt make sense... you cant have a plane with a fov >= math.pi
    assert args.fov < math.pi

    # the ankathete is important to 'normalize' the directions according to the resolution and fov
    # distance to plane
    ankathete = args.width / (2.0 * math.tan(args.fov / 2.0))

    ##############################
    # faster version with cv2.remap
    # this is actually remapping the whole image at once
    # remap needs: sourceimage (the panorama), a matrix with the x coords for the remap (map_x) and a matrix with the y coords for the remap (map_y)
    # so the value of the remapped image is like: remapped_image[y, x] = source_image[map_y[y, x], map_x[y, x]]
    # to make it fast you want to creat the whole map_x and map_y without any for loops
    ##############################

    # get the shift of the pixel coordinate from the middle of the view (as a matrix, so for every pixel at the same time)
    shift_y = 0.5 * args.height - np.repeat(np.arange(args.height, dtype=np.float32).reshape(-1,1), args.width, axis=1)
    shift_x = -0.5 * args.width + np.repeat(np.arange(args.width, dtype=np.float32).reshape(1,-1), args.height, axis=0)
    print('shift_x=', shift_x)
    print('shift_y=', shift_y)
    # compute the direction vector for EACH pixel simultaneously!
    # -shift_x in image plane to modify mirror
    # +shift_y in image plane to modify mirror, because shift_y is reverser to v-direction und hat minus-.
    # --shift_y = +shift_y
    # 3d coordinate of image plane
    direction_x = ankathete * direction[0] - shift_x * plane_x_direction_normalized[0] + shift_y * plane_y_direction_normalized[0]
    direction_y = ankathete * direction[1] - shift_x * plane_x_direction_normalized[1] + shift_y * plane_y_direction_normalized[1]
    direction_z = ankathete * direction[2] - shift_x * plane_x_direction_normalized[2] + shift_y * plane_y_direction_normalized[2]

    # convert the direction vector of each pixel into spherical coordinates of ODS
    length = np.sqrt(np.sum(np.stack((np.power(direction_x, 2), np.power(direction_y, 2), np.power(direction_z, 2))), axis=0))
    direction_theta = np.arccos(direction_y / length)                 # [0, math.pi] from positive y-axis to r
    # arctan2 return range [-pi, pi], 比值符号加在第一个输入值
    # + math.pi modify to 360Lib, phi from x counterclockwise to z
    direction_phi = np.arctan2(-direction_z, direction_x) + math.pi   # [0, 2*math.pi]
    # direction_phi = np.arctan(direction_z / -direction_x)# + math.pi  arctan return range [-pi/2, pi/2]

    # convert the spherical coordinates into pixel lookups in the ERP
    map_x_fast = direction_phi / (2*math.pi) * panowidth
    map_y_fast = direction_theta / math.pi * panoheight
    print('x', map_x_fast.shape)
    # remap the image (and save it), you can even choose an interpolation method <3
    out = cv2.remap(panorama, map_x_fast, map_y_fast, cv2.INTER_LINEAR)
    # cv2.imwrite(args.outputpath, cv2.remap(panorama, map_x_fast, map_y_fast, cv2.INTER_LINEAR))
    time_elapsed = time.time() - since
    print('Total complete in {:.0f}s {:.0f}ms'.format(
        modf(time_elapsed)[1], modf(time_elapsed)[0] * 1000))
    cv2.imshow('imgOut', out)
    # myRemap(panorama, map_x_fast, map_y_fast)
    cv2.waitKey(0)

