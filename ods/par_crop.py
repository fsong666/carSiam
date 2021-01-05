from os.path import join, isdir
from os import listdir, mkdir, makedirs
import xml.etree.ElementTree as ET
from concurrent import futures
import glob
import cv2
import numpy as np
import sys

ods_base_path = './dataset'
ann_base_path = join(ods_base_path, 'Annotations/train/')


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


def crop_hwc(image, bbox, out_sz, padding=(0, 0, 0)):
    a = (out_sz-1) / (bbox[2]-bbox[0])
    b = (out_sz-1) / (bbox[3]-bbox[1])
    c = -a * bbox[0]
    d = -b * bbox[1]
    mapping = np.array([[a, 0, c],
                        [0, b, d]]).astype(np.float)
    crop = cv2.warpAffine(image, mapping, (out_sz, out_sz), borderMode=cv2.BORDER_CONSTANT, borderValue=padding)
    return crop


def pos_s_2_bbox(pos, s):
    return [pos[0]-s/2, pos[1]-s/2, pos[0]+s/2, pos[1]+s/2]


def crop_like_SiamFC(image, bbox, context_amount=0.5, exemplar_size=127, instanc_size=255, padding=(0, 0, 0)):
    target_pos = [(bbox[2]+bbox[0])/2., (bbox[3]+bbox[1])/2.]
    target_size = [bbox[2]-bbox[0], bbox[3]-bbox[1]]
    wc_z = target_size[1] + context_amount * sum(target_size)
    hc_z = target_size[0] + context_amount * sum(target_size)
    s_z = np.sqrt(wc_z * hc_z)
    scale_z = exemplar_size / s_z
    d_search = (instanc_size - exemplar_size) / 2
    pad = d_search / scale_z
    s_x = s_z + 2 * pad

    z = crop_hwc(image, pos_s_2_bbox(target_pos, s_z), exemplar_size, padding)
    x = crop_hwc(image, pos_s_2_bbox(target_pos, s_x), instanc_size, padding)
    return z, x


def crop_video(video, crop_path, instanc_size):
    print('load ', video)
    video_crop_base_path = join(crop_path, video)
    if not isdir(video_crop_base_path): makedirs(video_crop_base_path)

    xmls = sorted(glob.glob(join(ann_base_path, video, '*.xml')))
    for idx, xml in enumerate(xmls):
        xmltree = ET.parse(xml)
        objects = xmltree.findall('object')
        filename = xmltree.findall('filename')[0].text.split('.')[0]

        imgIn = cv2.imread(xml.replace('xml', 'jpg').replace('Annotations', 'Data'))
        if imgIn is None:
            print('imgIn None:', xml)
        avg_chans = np.mean(imgIn, axis=(0, 1))
        for object_iter in objects:
            trackid = int(object_iter.find('trackid').text)
            bndbox = object_iter.find('bndbox')
            # two vertexes
            bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]

            z, x = crop_like_SiamFC(imgIn, bbox, instanc_size=instanc_size, padding=avg_chans)
            cv2.imwrite(join(video_crop_base_path, '{:06d}.{:02d}.x.jpg'.format(int(filename), trackid)), x)


def main(instanc_size=511, num_threads=6):
    crop_path = './crop{:d}'.format(instanc_size)
    if not isdir(crop_path): mkdir(crop_path)

    videos = sorted(listdir(ann_base_path))
    print(videos)
    n_videos = len(videos)
    with futures.ProcessPoolExecutor(max_workers=num_threads) as executor:
        fs = [executor.submit(crop_video, video, crop_path, instanc_size) for video in videos]
        for i, f in enumerate(futures.as_completed(fs)):
            printProgress(i, n_videos, prefix='ods', suffix='Done ', barLength=40)


if __name__ == '__main__':
    main()

