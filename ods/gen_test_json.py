import json
import xml.etree.ElementTree as ET
from os import listdir, mkdir
from os.path import join, isdir
import glob
import random

ods_base_path = './dataset'
ann_base_path = join(ods_base_path, 'Annotations/train/')


def run_gen():
    dicts = dict()
    videos = sorted(listdir(ann_base_path))

    for video in videos:
        video_value = dict()
        img_names = []
        gt_rect = []
        xmls = sorted(glob.glob(join(ann_base_path, video, '*.xml')))
        video = video.replace('train', 'test')
        # selected = random.sample(range(1, 600), 100)
        for idf, xml in enumerate(xmls):  # xml->frame, idf. 目录里的绝对索引
            if idf < 200 or idf > 500:
                continue
            frame = ET.parse(xml)
            filename = frame.findall('filename')[0].text
            if len(frame.findall('object')) < 1:
                print('{}.xml hat no object'.format(filename.split('.')[0]))
                return

            obj = frame.findall('object')[0]  # single trackid
            bndbox = obj.find('bndbox')
            bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                    int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
            bbox[2] -= bbox[0]  # w
            bbox[3] -= bbox[1]  # h
            video_value["video_dir"] = video
            if idf == 200:
                video_value["init_rect"] = bbox
            img_names.append(join(video, filename))
            gt_rect.append(bbox)
        video_value["img_names"] = img_names
        video_value["gt_rect"] = gt_rect
        dicts[video] = video_value

    # test_path = '../testing_dataset/ODS'
    test_path = '/home/sf/Documents/github_proj/carSiam/ods/dataset/Data/test'
    if not isdir(test_path): mkdir(test_path)
    ods = join(test_path, 'ODS.json')
    json.dump(dicts, open(ods, 'w'), indent=4, sort_keys=True)
    print('done!')


if __name__ == '__main__':
    run_gen()

