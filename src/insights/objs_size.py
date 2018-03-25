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


def _calc_sizes(objects, label=""):
    sizes = []
    for obj in objects:
        if label != "" and obj.find('name').text != label:
            continue
        xmin = float(obj.find('xmin').text)
        ymin = float(obj.find('ymin').text)
        xmax = float(obj.find('xmax').text)
        ymax = float(obj.find('ymax').text)
        area = round((xmax - xmin + 1) * (ymax - ymin + 1))
        sizes.append(math.sqrt(area))
    return sizes


def count_objs_size_per_image(data, label=""):
    return [size for xml in data for size in _calc_sizes(xml.findAll('object'), label)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loads aerial annotations')
    parser.add_argument('--data', dest='data_path', type=str,
                        default='../faster-rcnn.pytorch/data',
                        help='data dir path')
    parser.add_argument('--csv_data', dest='csv_data_path', type=str,
                        default='../../docs/charts/raw data/',
                        help='csv data dir path')

    args = parser.parse_args()
    # if AerialData.is_cached(args.data_path):
    #     data = AerialData.from_cache(args.data_path)
    # else:
    #     raise RuntimeError('AerialData is not cached, parse them first')
    ann_path = os.path.join(args.data_path, 'VOCdevkit2007/VOC2007/Annotations')
    data = pascal_xmls_to_dicts(ann_path)
    objs_size_per_image = count_objs_size_per_image(data)
    print('Max object sqrt(area): ', max(objs_size_per_image))
    print('Avg object sqrt(area): ', sum(objs_size_per_image) / len(objs_size_per_image))
    objs_size_per_image_without_poles = [size for size in objs_size_per_image if size != 1]
    print('Avg object sqrt(area) without utility poles: ',
          sum(objs_size_per_image_without_poles) / len(objs_size_per_image_without_poles))
    print('Min objects sqrt(area): ', min(objs_size_per_image))

    objs_size_per_image_per_large_vehicle = (count_objs_size_per_image(data, 'large vehicle'))
    objs_size_per_image_per_large_vehicle.sort()
    objs_size_per_image_per_small_vehicle = count_objs_size_per_image(data, 'small vehicle')
    objs_size_per_image_per_small_vehicle.sort()
    objs_size_per_image_per_solar_panel = count_objs_size_per_image(data, 'solar panel')
    objs_size_per_image_per_solar_panel.sort()

    final_csv = 'large vehicle, small vehicle, solar panel' + '\n'
    for i in range(max(len(objs_size_per_image_per_large_vehicle), len(objs_size_per_image_per_small_vehicle), len(objs_size_per_image_per_solar_panel))):
        row = '%s,%s,%s' % (
            objs_size_per_image_per_large_vehicle[i] if i < len(objs_size_per_image_per_large_vehicle) else '',
            objs_size_per_image_per_small_vehicle[i] if i < len(objs_size_per_image_per_small_vehicle) else '',
            objs_size_per_image_per_solar_panel[i] if i < len(objs_size_per_image_per_solar_panel) else '',
        )
        final_csv += row + '\n'
    with open(args.csv_data_path + 'Labels Size Distributions.csv', 'w+') as csv_file:
        csv_file.write(final_csv)
    print('Wrote data to csv data dir path')