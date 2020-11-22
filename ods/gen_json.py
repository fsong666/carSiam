import json
import numpy as np
import xml.etree.ElementTree as ET
from os import listdir, mkdir, makedirs
from os.path import join, isdir
import glob


ods_base_path = './dataset'
ann_base_path = join(ods_base_path, 'Annotations/train/')


def run_gen():
    snippets = dict()
    n_snippets = 0
    n_videos = 0
    origin_frame_start = 0

    videos = sorted(listdir(ann_base_path))

    for video in videos:
        n_videos += 1

        id_set = []
        store = dict()

        xmls = sorted(glob.glob(join(ann_base_path, video, '*.xml')))
        for idf, xml in enumerate(xmls):  # xml->frame, idf. 目录里的绝对索引
            xmltree = ET.parse(xml)
            if idf == 0:
                origin_frame_start = int(xmltree.findall('filename')[0].text.split('.')[0])

            origin_frame_id = int(xmltree.findall('filename')[0].text.split('.')[0]) - origin_frame_start
            if origin_frame_id not in id_set:
                id_set.append(origin_frame_id)
                store[origin_frame_id] = idf

        if len(id_set) > 0:
            snippets[video] = dict()
        else:
            continue

        origin_frame_ids = sorted(id_set)

        sequences = np.split(origin_frame_ids, np.array(np.where(np.diff(origin_frame_ids) > 1)[0]) + 1)
        sequences = [s for s in sequences if len(s) > 1]
        selected = 0
        if len(sequences) <= 0:
            continue
        # seq = sequences[-1]w
        snippet = dict()
        for seq in sequences:  # 多个连续帧段
            for origin_frame_id in seq:  # 记录所有连续段的帧
                frame_id = store[origin_frame_id]
                frame = ET.parse(xmls[frame_id])
                filename = frame.findall('filename')[0].text
                if len(frame.findall('object')) < 1:
                    print('{}.xml hat no object'.format(filename.split('.')[0]))
                    return
                o = frame.findall('object')[0]
                bndbox = o.find('bndbox')
                bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                        int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                filename = frame.findall('filename')[0].text.split('.')[0]  # '000600.00.x.jpg'[:-4]
                snippet[filename] = bbox

        snippets[video]['{:02d}'.format(selected)] = snippet
        n_snippets += 1
        print('video: {:d} snippets_num: {:d}'.format(n_videos, n_snippets))

    train = {k: v for (k, v) in snippets.items() if 'train' in k}
    val = {k: v for (k, v) in snippets.items() if 'val' in k}

    json.dump(train, open('train.json', 'w'), indent=4, sort_keys=True)
    # json.dump(val, open('val.json', 'w'), indent=4, sort_keys=True)
    print('done!')


def main(instanc_size=511):
    snippets = dict()
    n_snippets = 0
    n_videos = 0

    videos = sorted(listdir(ann_base_path))

    for video in videos:
        n_videos += 1

        id_set = []
        id_frames = [[]] * 60
        xmls = sorted(glob.glob(join(ann_base_path, video, '*.xml')))
        for idf, xml in enumerate(xmls):  # xml->frame, idf. 目录里的绝对索引
            xmltree = ET.parse(xml)
            objects = xmltree.findall('object')
            filename = xmltree.findall('filename')[0].text.split('.')[0]
            for object_iter in objects:
                trackid = int(object_iter.find('trackid').text)
                if trackid not in id_set:
                    id_set.append(trackid)
                    id_frames[trackid] = []
                id_frames[trackid].append(idf)

        if len(id_set) > 0:
            snippets[video] = dict()

        for selected in id_set:
            frame_ids = sorted(id_frames[selected])
            sequences = np.split(frame_ids, np.array(np.where(np.diff(frame_ids) > 1)[0]) + 1)
            sequences = [s for s in sequences if len(s) > 1]
            print(video, ': ', sequences)
            # seq = sequences[-1]
            for seq in sequences:  # 多个连续帧段
                snippet = dict()
                for frame_id in seq:
                    frame = ET.parse(xmls[frame_id])
                    o = None
                    for obj in frame.findall('object'):
                        trackid = int(obj.find('trackid').text)
                        if trackid == selected:
                            o = obj
                            continue
                    assert o is not None
                    bndbox = o.find('bndbox')
                    bbox = [int(bndbox.find('xmin').text), int(bndbox.find('ymin').text),
                            int(bndbox.find('xmax').text), int(bndbox.find('ymax').text)]
                    filename = frame.findall('filename')[0].text[:-4]  # '000600.00.x.jpg'[:-4]
                    snippet[filename] = bbox
                print(snippet)
                snippets[video]['{:02d}'.format(selected)] = snippet
                n_snippets += 1

        print('video: {:d} snippets_num: {:d}'.format(n_videos, n_snippets))

    train = {k: v for (k, v) in snippets.items() if 'train' in k}
    val = {k: v for (k, v) in snippets.items() if 'val' in k}

    # json.dump(train, open('train.json', 'w'), indent=4, sort_keys=True)
    # json.dump(val, open('val.json', 'w'), indent=4, sort_keys=True)
    print('done!')


if __name__ == '__main__':
    # main()
    run_gen()

