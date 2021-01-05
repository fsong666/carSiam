import cv2
import numpy as np
# from psd_tools import PSDImage


def sift(src, dst, depth1, depth2):
    # img1 = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
    # img2 = cv2.imread(dst, cv2.IMREAD_GRAYSCALE)
    # img1 = cv2.imread(src)
    # img2 = cv2.imread(dst)
    img1 = src
    img2 = dst

    # 3) SIFT特征计算
    # sift = cv2.xfeatures2d.SIFT_create()
    sift = cv2.SIFT_create()

    keypt1, des1 = sift.detectAndCompute(img1, None)
    keypt2, des2 = sift.detectAndCompute(img2, None)

    # 4) Flann特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    goodMatch = []
    for m, n in matches:
        # goodMatch是经过筛选的优质配对，如果2个配对中第一匹配的距离小于第二匹配的距离的1/2，基本可以说明这个第一配对是两幅图像中独特的，不重复的特征点,可以保留。
        if m.distance < 0.50*n.distance:
            goodMatch.append(m)
            # print(keypt1[m.queryIdx].pt, ' ', keypt2[m.queryIdx].pt)
    # >10 240
    # 7 <= disparity < 10 256
    # 4 <= disparity < 7 223
    # 1 <= disparity < 4 152
    # 1 <= disparity < 7 180
    # 1 <= disparity < 6 175
    # 2 <= disparity < 6 187
    base = 0
    best_start = 0
    best_end = -1
    depth_list = []
    disparity_list = []
    index = []
    for idx, m in enumerate(goodMatch[best_start:best_end]):
        pt1 = keypt1[m.queryIdx].pt
        pt2 = keypt2[m.trainIdx].pt
        disparity = abs(pt1[0] - pt2[0])
        vu_pt1 = (round(pt1[1]), round(pt1[0]))  # (y,x)
        vu_pt2 = (round(pt2[1]), round(pt2[0]))
        depth = (depth1[vu_pt1] + depth2[vu_pt2]) * 0.5
        if 2 <= disparity < 6:
            baseline = depth * disparity
            base += baseline
            index.append(idx)
            depth_list.append(depth)
            disparity_list.append(disparity)
            # print('depth1={} depth2={}'.format(depth1[vu_pt1], depth2[vu_pt2]))
            print('{}:{} -- {} - {} - {} -> {}'.format(idx, pt1, pt2, disparity, depth, baseline))
            # cv2.circle(img1, (int(pt1[0]), int(pt1[1])), 3, (0, 0, 255-idx*50), 2)
            # cv2.circle(img2, (int(pt2[0]), int(pt2[1])), 3, (0, 0, 255-idx*50), 2)
    num = len(index)
    if num < 1:
        return
    baseline = base / num
    pred_disparity = baseline / np.array(depth_list)
    sums = abs(pred_disparity - np.array(disparity_list)).sum()
    loss = abs(pred_disparity - np.array(disparity_list)).mean()
    print('baseline:', baseline)
    print('depth:\n', depth_list)
    print('pred_disparity:\n', pred_disparity)
    print('diff_disparity:\n', pred_disparity - np.array(disparity_list))

    print('sum:\n', sums)
    print('loss:\n', loss)
    cv2.imshow('img1', img1)
    cv2.imshow('img2', img2)

    # 增加一个维度
    goodMatch = np.expand_dims(goodMatch, 1)
    img_out = cv2.drawMatchesKnn(img1, keypt1, img2, keypt2, goodMatch[index], None, flags=2)

    cv2.imshow('image', img_out)#展示图片
    cv2.waitKey(0)#等待按键按下
    cv2.destroyAllWindows()#清除所有窗口
    # cv2.imwrite('./test_img/front_right.jpg', img_out)



def test():
    path = './test_img/test_stereo.png'
    img = cv2.imread(path)
    (h, w) = img.shape[:2]
    cx = w // 2
    cy = h // 2
    for x in range(0, w, 100):
        cv2.line(img, (x, 0), (x, h-1), (0, 255, 0), 1)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    # cv2.imwrite('./test_img/test_stereo.png', img)


def gen_two():
    imgIn1 = './test_img/0600left.png'
    imgIn2 = './test_img/0600right.png'

    imgDepth1 = './test_img/depthLeft.png'
    imgDepth2 = './test_img/depthRight.png'
    imgDepth1 = cv2.imread(imgDepth1, cv2.IMREAD_GRAYSCALE)
    imgDepth2 = cv2.imread(imgDepth2, cv2.IMREAD_GRAYSCALE)

    imgIn1 = cv2.imread(imgIn1)
    imgIn2 = cv2.imread(imgIn2)
    cx = imgIn1.shape[1]//2  # 600
    edge = cx // 4

    viewer1 = [cx-100, 400]
    viewer2 = [cx-100, 400]
    print('origin viewer:', viewer1)

    cv2.circle(imgIn1, tuple(viewer1), 3, (0, 255, 0), 3)
    cv2.circle(imgIn2, tuple(viewer2), 3, (0, 255, 0), 3)

    pt1 = [405-edge, 287-50]
    pt2 = [405+edge, 287+edge+50]

    print(pt1)
    print(pt2)
    imgOut1 = imgIn1[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    imgOut2 = imgIn2[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    imgDepth1 = imgDepth1[pt1[1]:pt2[1], pt1[0]:pt2[0]]
    imgDepth2 = imgDepth2[pt1[1]:pt2[1], pt1[0]:pt2[0]]

    cv2.imshow('imgDepth1', imgDepth1)
    cv2.imshow('imgDepth2', imgDepth2)
    # cv2.imshow('imgIn2', imgIn2)
    # cv2.imshow('imgOut2', imgOut2)
    # cv2.imwrite('./test_img/test_left.png', imgOut1)
    # cv2.imwrite('./test_img/test_right.png', imgOut2)

    sift(imgOut1, imgOut2, imgDepth1, imgDepth2)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":
    # img1 = './test_img/000600.jpg'
    # img2 = './test_img/000650.jpg'

    img1 = './test_img/test_left.png'
    img2 = './test_img/test_right.png'
    # sift(img1, img2)
    # test()
    gen_two()