import argparse
import collections
import random
import os
import xml.etree.cElementTree as ET
import shutil
import _init_paths
from load_annotations import AerialData
from utils.logger import Logger
import json


class VOTT:
    def __init__(self, data_dir_path, labels, limit_resolution):
        self.data_dir_path = data_dir_path
        self.labels = labels
        self.limit_resolution = limit_resolution
        self.root_dirname = 'VOTT'
        self.train_fname = 'Images with objects.json'

    def _gen_json_file(self, data):
        fpath = os.path.join(self.data_dir_path, self.root_dirname, self.train_fname)
        data = json.dumps(data, indent=4)
        with open(fpath, 'w') as json_file:
            json_file.write(data)
            Logger.log('Generated %s file' % self.train_fname)

    def _gen_dict_format(self, data):
        dict_result = {
            'frames': {},
            'framerate': 1,
            'inputTags': self.labels,
            'suggestiontype': 'track',
            'scd': 'false',
            'visitedFrames': sorted([image_id for image_id in data])
        }
        id_counter = 0
        for (i, image_id) in enumerate(data):
            dict_result['frames'][i] = []
            image = data[image_id]

            for (j, obj) in enumerate(image.objects):
                obj_dict = {'id': id_counter}
                id_counter += 1
                obj_dict['name'] = obj.label
                obj_dict['tags'] = [obj.label]
                obj_dict['height'] = image.height
                obj_dict['width'] = image.width
                xmin, ymin, xmax, ymax = self._obb_to_pascal_hbb(obj.bounding_box.get_points())
                assert xmin <= xmax < image.width and ymin <= ymax < image.height
                x1, y1, x2, y2 = str(xmin), str(ymin), str(xmax), str(ymax)
                obj_dict['x1'] = x1
                obj_dict['y1'] = y1
                obj_dict['x2'] = x2
                obj_dict['y2'] = y2
                dict_result['frames'][i].append(obj_dict)
        dict_result['frames'] = collections.OrderedDict(sorted(dict_result['frames'].items()))
        return dict_result

    def _create_dirpath_if_not_exist(self, dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    def _rm_dir(self, dir_path):
        for f in os.listdir(dir_path):
            file_path = os.path.join(dir_path, f)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    def _assert_dirs(self):
        dir_path = os.path.join(self.data_dir_path, self.root_dirname)
        self._create_dirpath_if_not_exist(dir_path)
        self._rm_dir(dir_path)

    def _obb_to_pascal_hbb(self, obb):
        xmin = min([float(x) for x, _ in obb])
        ymin = min([float(y) for _, y in obb])
        xmax = max([float(x) for x, _ in obb])
        ymax = max([float(y) for _, y in obb])
        return float(xmin), float(ymin), float(xmax), float(ymax)

    '''
    Remove:
    1. Untagged and No objects images
    2. High resolution images
    3. Utility poles
    '''

    def _filter_images(self, data):
        initial_len = len(data)
        filtered_data = data
        Logger.log('Data contains %d images' % initial_len)

        # Remove utility poles
        # if self.omit_utility_poles:
        #     omitted_images = 0
        #     for img_id in filtered_data:
        #         original_objects_len = len(filtered_data[img_id].objects)
        #         filtered_data[img_id].objects = [obj for obj in filtered_data[img_id].objects if
        #                                          obj.label != 'utility pole']
        #         if len(filtered_data[img_id].objects) == 0 and original_objects_len > 0:
        #             filtered_data[img_id].category = 'No objects'
        #             omitted_images += 1
        #     Logger.log('Filtered utility poles from images, %d images was marked with \'No objects\'' % omitted_images)

        # Remove untagged and no object images
        filtered_data = {img_id: filtered_data[img_id] for img_id in filtered_data if
                         filtered_data[img_id].category not in ['Untagged', 'No objects']}
        Logger.log('Removed %d untagged and no object images' % (initial_len - len(filtered_data)))
        initial_len = len(filtered_data)

        # Remove high resolution images
        if self.limit_resolution:
            filtered_data = {img_id: filtered_data[img_id] for img_id in filtered_data if
                             max(filtered_data[img_id].width, filtered_data[img_id].height) <= self.max_resolution}
            Logger.log('Removed %d high resolution images' % (initial_len - len(filtered_data)))
            initial_len = len(filtered_data)

        Logger.log('Final images amount is %d' % len(filtered_data))

        return filtered_data

    def _fix_single_point(self, img_id, pt, width, height):
        if float(pt[0]) < 0:
            Logger.log('Image %s contained negative point %s' % (img_id, pt[0]))
            pt = (str(0), pt[1])
        elif float(pt[0]) >= width:
            Logger.log('Image %s contained out of width point: %s >= %d' % (img_id, pt[0], width))
            pt = (str(width - 1), pt[1])

        if float(pt[1]) < 0:
            Logger.log('Image %s contained negative point %s' % (img_id, pt[1]))
            pt = (pt[0], str(0))
        elif float(pt[1]) >= height:
            Logger.log('Image %s contained out of height point: %s >= %d' % (img_id, pt[1], height))
            pt = (pt[0], str(height - 1))

        return pt

    def _fix_exceeding_bounding_boxes(self, data):
        # Remove images whose coordinates exceed the image
        for img_id in data:
            img = data[img_id]
            image_id = img.image_id
            height = img.height
            width = img.width
            for obj in img.objects:
                p1, p2, p3, p4 = obj.bounding_box.pt1, obj.bounding_box.pt2, obj.bounding_box.pt3, obj.bounding_box.pt4
                obj.bounding_box.pt1 = self._fix_single_point(image_id, p1, width, height)
                obj.bounding_box.pt2 = self._fix_single_point(image_id, p2, width, height)
                obj.bounding_box.pt3 = self._fix_single_point(image_id, p3, width, height)
                obj.bounding_box.pt4 = self._fix_single_point(image_id, p4, width, height)

    def convert(self, aerial_data):
        filtered_data = self._filter_images(aerial_data.data)
        self._fix_exceeding_bounding_boxes(filtered_data)
        self._assert_dirs()
        data = self._gen_dict_format(filtered_data)
        self._gen_json_file(data)
        return filtered_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loads aerial annotations')
    parser.add_argument('--train_tags', dest='train_tags', type=str,
                        default='../../Detecting And Classifying Objects In Aerial Imagery/Train/CSV/Train_tags.csv',
                        help='train tags file path')
    parser.add_argument('--train_details', dest='train_details', type=str,
                        default='../../Detecting And Classifying Objects In Aerial Imagery/Train/CSV/Train_imagery_details.csv',
                        help='train tags file path')
    parser.add_argument('--images_dir', dest='images_dir', type=str,
                        default='../../Detecting And Classifying Objects In Aerial Imagery/Train/Imagery data',
                        help='Images file path')
    parser.add_argument('--train_split_percent', dest='train_split_percent', type=int,
                        default=90,
                        help='percentage of train from whole dataset', choices=range(1, 100))
    parser.add_argument('--data', dest='data_path', type=str,
                        default='../faster-rcnn.pytorch/data',
                        help='data dir path')
    parser.add_argument('--cached-aerial-data', dest='load_cached_aerial_data', type=bool,
                        default=True,
                        help='whether to load cached aerial data or generate from scratch')
    parser.add_argument('--limit-resolution', dest='limit_resolution', type=bool,
                        default=False,
                        help='whether to limit loaded images resolutions')
    parser.add_argument('--max-resolution', dest='max_resolution', type=int,
                        default=1000,
                        help='max resolution of images', choices=range(1, 5000))

    args = parser.parse_args()
    if args.load_cached_aerial_data and AerialData.is_cached(args.data_path):
        data = AerialData.from_cache(args.data_path)
    else:
        data = AerialData(args.data_path)
        data.load(args.train_details, args.train_tags, args.images_dir)
    vott = VOTT(args.data_path, ['large vehicle', 'small vehicle', 'solar panel', 'utility pole'], args.limit_resolution)
    data = vott.convert(data)
    print('Converted images: (', ' '.join([img.fname for img_id, img in data.items()]), ')')