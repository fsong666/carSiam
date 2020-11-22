from os.path import join, isdir
from os import listdir, mkdir, makedirs
import xml.etree.ElementTree as ET
from concurrent import futures
import glob
import cv2
from ods.center_cube import ViewerCUbe
import sys

ods_base_path = './dataset'
originAnn_base_path = join(ods_base_path, 'Annotations/origin_data/')


def printProgress(iteration, total, prefix='', suffix='', decimals=1, barLength=100):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        barLength   - Optional  : character length of bar (Int)
    """
    formatStr       = "{0:." + str(decimals) + "f}"
    percents        = formatStr.format(100 * (iteration / float(total)))
    filledLength    = int(round(barLength * iteration / float(total)))
    bar             = '' * filledLength + '-' * (barLength - filledLength)
    sys.stdout.write('\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)),
    if iteration == total:
        sys.stdout.write('\x1b[2K\r')
    sys.stdout.flush()


def crop_cubeVideo(video, crop_path=None):
    xmls = sorted(glob.glob(join(originAnn_base_path, video, '*.xml')))
    viewerCube = None
    ann_videos = []
    for idx, xml in enumerate(xmls):
        xmltree = ET.parse(xml)
        # print(xml)
        objects = xmltree.findall('object')
        filename = xmltree.findall('filename')[0].text.split('.')[0]
        imgIn = cv2.imread(xml.replace('xml', 'png').replace('Annotations', 'Data'))

        if idx == 0:
            viewerCube = ViewerCUbe(imgIn.shape)

        for object_iter in objects:
            trackid = int(object_iter.find('trackid').text)
            bndbox = object_iter.find('bndbox')
            # two vertexes
            bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
            center = [(bbox[2] + bbox[0]) / 2., (bbox[3] + bbox[1]) / 2.]

            imgOut = viewerCube.reversToCube(imgIn, center)

            video_crop_base_path = join(crop_path, '{}_train_{:02d}'.format(video, trackid))
            if not isdir(video_crop_base_path):
                makedirs(video_crop_base_path)
                ann_videos.append(video_crop_base_path.replace('Data', 'Annotations'))

            cv2.imwrite(join(video_crop_base_path, '{:06d}.jpg'.format(int(filename))), imgOut)

    # generate dirs of annotation for each cube video
    for ann_path in ann_videos:
        if not isdir(ann_path):
            makedirs(ann_path)


def show(imgIn, bbox, imgOut):
    cv2.rectangle(imgIn, (bbox[0], bbox[1]),
                  (bbox[2], bbox[3]),
                  (0, 255, 0), 2)
    cv2.imshow('img', imgIn)
    cv2.imshow('cube', imgOut)
    cv2.waitKey(0)


def gen_centerCube(num_threads=4):
    crop_path = join(ods_base_path, 'Data/train')
    if not isdir(crop_path): mkdir(crop_path)

    videos = sorted(listdir(originAnn_base_path))
    print(videos)
    n_videos = len(videos)
    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        fs = [executor.submit(crop_cubeVideo, video, crop_path) for video in videos]
        for i, f in enumerate(futures.as_completed(fs)):
            printProgress(i, n_videos, prefix='cube ods', suffix='Done ', barLength=40)


if __name__ == '__main__':
    gen_centerCube()