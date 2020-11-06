# -*- coding: utf-8 -*-

from __future__ import print_function, division
import numpy as np
import cv2
from scipy.interpolate import interp2d

""" ------------------ parameters ----------------- """
path = '1158.png'
# path = 'img_fish.png'
src = cv2.imread(path, -1) / 255.  # source fisheye image
h, w, c = src.shape

rx, ry = 1024., 1024.  # center of the fisheye circle, (0, 0) for bottom left corner
R = 1024.  # radius of the fisheye circle

fov = 210 * np.pi / 180.  # field of view of the fishye circle
W, H = 600, 300  # size of the output panorama

interp_method = 'cubic'  # must in ['linear', 'cubic', 'quintic']
upsample = 2  # factor of super-sampling anti-aliasing

yaw, pitch, roll = 0, 0, 0  # degree

""" ----------------- processing -------------------- """

up_W, up_H = W * upsample, H * upsample

yaw = yaw * np.pi / 180.  # rotate around z, angle from x+ to y+
yaw_m = np.array([[np.cos(-yaw), -np.sin(-yaw), 0], [np.sin(-yaw), np.cos(-yaw), 0], [0, 0, 1]])

pitch = pitch * np.pi / 180.  # rotate around x, angle from y+ to z+
pitch_m = np.array([[1, 0, 0], [0, np.cos(-pitch), -np.sin(-pitch)], [0, np.sin(-pitch), np.cos(-pitch)]])

roll = roll * np.pi / 180.  # rotate around y, angle from x+ to z+
roll_m = np.array([[np.cos(-roll), 0, -np.sin(-roll)], [0, 1, 0], [np.sin(-roll), 0, np.cos(-roll)]])

affine_m = yaw_m.dot(pitch_m).dot(roll_m)

# by default, the horizontal FOV is [-pi, pi], the verticle FOV is [-pi/2, pi/2]
phi = np.linspace(1, -1, up_H, endpoint=False) * np.pi * 0.5  # latitude of the equirectangular
theta = np.linspace(-1, 1, up_W, endpoint=False) * np.pi  # longitude of the equirectangular

theta, phi = np.meshgrid(theta, phi)

x = np.cos(phi) * np.sin(theta)  # 3D space x
y = np.cos(phi) * np.cos(theta)  # 3D space y
z = np.sin(phi)  # 3D space z

# yaw, pitch, roll inverse affine tranformation
xyz = np.vstack([mat.reshape(-1) for mat in [x, y, z]])
no_rpy_xyz = affine_m.dot(xyz)

x, y, z = [coord.reshape([up_H, up_W]) for coord in no_rpy_xyz]

phi_f = np.arctan2(np.sqrt(x * x + z * z), (y + 1e-12))  # incident angle of the fisheye
theta_f = np.arctan2(z, (x + 1e-12))  # projection angle of the fisheye plane

r_f = phi_f * 2 * R / fov  # equidistant model, where r = C * theta
x_f = rx + r_f * np.cos(theta_f)  # x coordinate on the fisheye plane
y_f = ry + r_f * np.sin(theta_f)  # y coordinate on the fisheye plane

dst = np.zeros([up_H, up_W, c])
for i in range(c):
    interp = interp2d(np.arange(w), np.arange(h)[::-1], src[:, :, i], interp_method, fill_value=0)
    for j in range(up_H):
        for k in range(up_W):
            if (x_f[j, k] < w and y_f[j, k] < h):
                dst[j, k, i] = interp(x_f[j, k], y_f[j, k])

dst = (np.clip(dst, 0, 1) * 255).astype('uint8')
if upsample > 1:
    dst = cv2.resize(dst, (W, H), cv2.INTER_CUBIC)
cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite('panorama.png', dst)

