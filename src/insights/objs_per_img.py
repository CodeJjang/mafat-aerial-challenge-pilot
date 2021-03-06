import argparse
import _init_paths
from converters.load_annotations import AerialData
from bs4 import BeautifulSoup
import os


def pascal_xmls_to_dicts(xmls_path):
    res = []
    for filename in os.listdir(xmls_path):
        if filename.endswith(".xml"):
            fpath = os.path.join(xmls_path, filename)
            with open(fpath, 'r') as xml_file:
                res.append(BeautifulSoup(xml_file.read(), 'lxml'))
    return res

def count_objs_per_image(data):
    return [len(xml.findAll('object')) for xml in data if len(xml.findAll('object')) > 0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loads aerial annotations')
    parser.add_argument('--data', dest='data_path', type=str,
                        default='../faster-rcnn.pytorch/data',
                        help='data dir path')

    args = parser.parse_args()
    # if AerialData.is_cached(args.data_path):
    #     data = AerialData.from_cache(args.data_path)
    # else:
    #     raise RuntimeError('AerialData is not cached, parse them first')
    ann_path = os.path.join(args.data_path, 'VOCdevkit2007/VOC2007/Annotations')
    data = pascal_xmls_to_dicts(ann_path)
    objs_per_img = count_objs_per_image(data)
    print('Max objects per image: ', max(objs_per_img))
    print('Avg objects per image: ', sum(objs_per_img)/len(objs_per_img))
    print('Min objects per image: ', min(objs_per_img))
