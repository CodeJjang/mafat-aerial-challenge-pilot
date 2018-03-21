import argparse
import _init_paths
from converters.load_annotations import AerialData
from bs4 import BeautifulSoup
import os
import math


def pascal_xmls_to_dicts(xmls_path):
    res = []
    for filename in os.listdir(xmls_path):
        if filename.endswith(".xml"):
            fpath = os.path.join(xmls_path, filename)
            with open(fpath, 'r') as xml_file:
                res.append(BeautifulSoup(xml_file.read(), 'lxml'))
    return res

def _calc_sizes(objects):
    sizes = []
    for obj in objects:
        xmin = float(obj.find('xmin').text)
        ymin = float(obj.find('ymin').text)
        xmax = float(obj.find('xmax').text)
        ymax = float(obj.find('ymax').text)
        area = round((xmax - xmin + 1) * (ymax - ymin + 1))
        sizes.append(math.sqrt(area))
    return sizes

def count_objs_size_per_image(data):
    return [size for xml in data for size in _calc_sizes(xml.findAll('object'))]


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
    objs_size_per_image = count_objs_size_per_image(data)
    print('Max object sqrt(area): ', max(objs_size_per_image))
    print('Avg object sqrt(area): ', sum(objs_size_per_image)/len(objs_size_per_image))
    print('Min objects sqrt(area): ', min(objs_size_per_image))