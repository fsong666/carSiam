import json
import cv2
import os

json_path = './train.json'
data_path = '/home/sf/Documents/github_proj/carSiam/ods/dataset/Data/train'
path_format = '{}.{}.{}.jpg'
path_format2 = '{}.jpg'


def test():
    with open(json_path, 'r') as f:
        meta_data = json.load(f)

    for video, tracks in meta_data.items():
        for trk, frames in tracks.items():
            for frm, bbox in frames.items():
                if not isinstance(bbox, dict):
                    if len(bbox) == 4:
                        x1, y1, x2, y2 = bbox
                        w, h = x2 - x1, y2 - y1
                    else:
                        w, h = bbox
                    if w <= 0 or h <= 0:  # remove w h <= 0
                        continue
                # image_path = os.path.join(data_path, video,
                #                           path_format.format(frm, trk, 'x'))
                image_path = os.path.join(data_path, video,
                                          path_format2.format(frm))
                img = cv2.imread(image_path)
                if img is None:
                    print('None:', image_path)
                cv2.rectangle(img,
                              (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]),
                              (0, 255, 0), 2)

                cv2.imshow('img', img)
                cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test()