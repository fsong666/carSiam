import cv2
import numpy as np
from math import pi




def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz - 1) / (bbox[2] - bbox[0])
    b = (out_sz - 1) / (bbox[3] - bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def transform(img, dst, f=8.2):
    h = img.shape[0]  # 600
    w = img.shape[1]  # 1200
    # R = min(h, w) // 2
    # f = 2 * R / pi
    h_c = h // 2
    w_c = w // 2
    print('h_c={}, w_c={}'.format(h_c, w_c))
    for y in range(h):
        dy = y - h_c
        for x in range(w):
            dx = x - w_c
            r = np.sqrt(dy**2 + dx**2)
            theta = r / f * pi / 180
            new_r = f * np.tan(theta)

            # phi = np.arctan2(dy, dx)
            u = new_r * dx / r
            v = new_r * dy / r

            u = 0 if np.isnan(u) else u
            v = 0 if np.isnan(v) else v

            u = int(u + w_c)
            v = int(v + h_c)
            u = max(0, min(w - 1, u))
            v = max(0, min(h - 1, v))

            if y == 100 and x == 100:
                print('x={}, u={}'.format(x, u))
            dst[v, u] = img[y, x]

    p1 = [w_c - 50, h_c - 50]
    p2 = [p1[0] + 100, p1[1] + 100]
    bbox = [p1[0], p1[1], p2[0], p2[1]]
    # cv2.rectangle(dst, (p1), p2, (0, 255, 0), thickness=2)
    dst = crop_hwc(dst, bbox, h)
    return dst


def transform2(img, dst, f=8):
    h = img.shape[0]
    w = img.shape[1]
    R = min(h, w) // 2
    f = 2 * R / pi
    h_c = h // 2
    w_c = w // 2
    print('h_c={}, w_c={}'.format(h_c, w_c))
    for y in range(h):
        dy = y - h_c
        for x in range(w):
            dx = x - w_c
            h0 = np.sqrt(dy ** 2 + dx ** 2)
            h1 = f * np.arctan2(h0, f)

            u = h1 * dx / h0
            v = h1 * dy / h0

            u = 0 if np.isnan(u) else u
            v = 0 if np.isnan(v) else v

            u = int(u + w_c)
            v = int(v + h_c)
            u = max(0, min(w - 1, u))
            v = max(0, min(h - 1, v))

            if y == 100 and x == 100:
                print('x={}, u={}'.format(x, u))
            dst[v, u] = img[y, x]

    return dst


def transform3(img):
    # 经度矫正法
    h = img.shape[0]
    w = img.shape[1]
    cy = h / 2
    cx = w / 2 #+ 200
    R = 150
    dst = np.zeros(img.shape)
    for v in range(h):
        for u in range(w):
            y = v
            x = np.sqrt(R ** 2 - (y - cy) ** 2) * (u - cx) / R + cx
            x = 0 if np.isnan(x) else x
            x = round(x)
            if R ** 2 - (y - cy) ** 2 < 0:
                continue
            dst[v, u] = img[y, x]

    return np.uint8(dst)


def transform4(img, dst, f=8.2):
    h = img.shape[0]  # 600
    w = img.shape[1]  # 1200
    h_c = h // 2
    w_c = w // 2
    print('h_c={}, w_c={}'.format(h_c, w_c))
    for y in range(h):
        phi = (0.5 - (y + 0.5) / h) * pi
        for x in range(w):

            theta = ((x + 0.5) / w - 0.5)* 2 * pi

            u = (2.0 * w * (theta + pi) / pi)
            v = (2.0 * h * (pi / 2 - phi) / pi)


            u = 0 if np.isnan(u) else u
            v = 0 if np.isnan(v) else v

            u = int(u + w_c)
            v = int(v + h_c)
            u = max(0, min(w - 1, u))
            v = max(0, min(h - 1, v))

            if y == 100 and x == 100:
                print('x={}, u={}'.format(x, u))
            dst[v, u] = img[y, x]
    return dst

def show(img, dst):
    cv2.imshow('img', img)
    cv2.imshow('dst', dst)
    cv2.waitKey(0)


def input_img():#
    path = '1158.png'
    # path = 'img_fish.jpg'
    img = cv2.imread(path)
    dst = np.zeros(img.shape, dtype=img.dtype)
    print('dst', dst.shape)
    dst = transform4(img, dst)
    # dst = transform3(img)
    show(img, dst)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    input_img()
