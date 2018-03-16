import argparse
import random
import os
import xml.etree.cElementTree as ET
import shutil
import _init_paths
from load_annotations import AerialData
from utils.logger import Logger


class PascalVoc:
    def __init__(self, data_dir_path, train_split_percent, limit_resolution, max_resolution):
        self.data_dir_path = data_dir_path
        self.train_split_percent = train_split_percent
        self.limit_resolution = limit_resolution
        self.max_resolution = max_resolution
        self.root_dirname = 'VOCdevkit2007'
        self.dataset_dirname = 'VOC2007'
        self.image_ids_dirname = 'ImageSets/Main'
        self.images_dirname = 'JPEGImages'
        self.annotations_dirname = 'Annotations'
        self.train_fname = 'train.txt'
        self.val_fname = 'val.txt'
        self.trainval_fname = 'trainval.txt'
        self.images_depth = 3

    def _gen_train_file(self, train_data):
        fpath = os.path.join(self.data_dir_path, self.root_dirname, self.dataset_dirname, self.image_ids_dirname,
                             self.train_fname)
        content = '\n'.join(train_data)
        with open(fpath, 'w') as train_file:
            train_file.write(content)
            Logger.log('Generated train file with ids')

    def _gen_val_file(self, val_data):
        fpath = os.path.join(self.data_dir_path, self.root_dirname, self.dataset_dirname, self.image_ids_dirname,
                             self.val_fname)
        content = '\n'.join(val_data)
        with open(fpath, 'w') as val_file:
            val_file.write(content)
            Logger.log('Generated val file with ids')

    def _gen_trainval_file(self, trainval_data):
        fpath = os.path.join(self.data_dir_path, self.root_dirname, self.dataset_dirname, self.image_ids_dirname,
                             self.trainval_fname)
        content = '\n'.join(trainval_data)
        with open(fpath, 'w') as trainval_file:
            trainval_file.write(content)
            Logger.log('Generated trainval file with ids')

    def _gen_annotations(self, data):
        dirpath = os.path.join(self.data_dir_path, self.root_dirname, self.dataset_dirname, self.annotations_dirname)
        for image_id in data:
            image = data[image_id]
            fpath = os.path.join(dirpath, image_id + '.xml')

            root = ET.Element("annotations")
            ET.SubElement(root, "folder").text = self.dataset_dirname
            ET.SubElement(root, "filename").text = image.fname

            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(image.width)
            ET.SubElement(size, "height").text = str(image.height)
            ET.SubElement(size, "depth").text = str(self.images_depth)

            ET.SubElement(root, "segmented").text = 0

            for obj in image.objects:
                object = ET.SubElement(root, "object")
                ET.SubElement(object, "name").text = obj.label
                ET.SubElement(object, "pose").text = 'Unspecified'
                ET.SubElement(object, "truncated").text = str(0)
                ET.SubElement(object, "difficult").text = str(0)
                bb = ET.SubElement(object, "bndbox")
                xmin, ymin, xmax, ymax = self._obb_to_pascal_hbb(obj.bounding_box.get_points())
                assert xmin <= xmax < image.width and ymin <= ymax < image.height
                ET.SubElement(bb, 'xmin').text, \
                ET.SubElement(bb, 'ymin').text, \
                ET.SubElement(bb, 'xmax').text, \
                ET.SubElement(bb, 'ymax').text = str(xmin), str(ymin), str(xmax), str(ymax)

            tree = ET.ElementTree(root)
            tree.write(fpath)
        Logger.log('Generated annotations')

    def _create_dirpath_if_not_exist(self, dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    def _rm_dir(self, dir_path):
        for f in os.listdir(dir_path):
            file_path = os.path.join(dir_path, f)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    def _assert_dirs(self):
        image_ids_dirs_path = os.path.join(self.data_dir_path, self.root_dirname, self.dataset_dirname,
                                           self.image_ids_dirname)
        images_dir_path = os.path.join(self.data_dir_path, self.root_dirname, self.dataset_dirname, self.images_dirname)
        annotations_dir_path = os.path.join(self.data_dir_path, self.root_dirname, self.dataset_dirname,
                                            self.annotations_dirname)
        self._create_dirpath_if_not_exist(image_ids_dirs_path)
        self._create_dirpath_if_not_exist(images_dir_path)
        self._create_dirpath_if_not_exist(annotations_dir_path)
        self._rm_dir(image_ids_dirs_path)
        self._rm_dir(annotations_dir_path)

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
    '''

    def _filter_images(self, data):
        initial_len = len(data)
        Logger.log('Data contains %d images' % initial_len)

        # Remove untagged and no object images
        filtered_data = {img_id: data[img_id] for img_id in data if
                         data[img_id].category not in ['Untagged', 'No objects']}
        Logger.log('Removed %d untagged and no object images' % (initial_len - len(filtered_data)))
        initial_len = len(filtered_data)

        # Remove high resolution images
        if self.limit_resolution:
            filtered_data = {img_id: filtered_data[img_id] for img_id in filtered_data if
                             max(filtered_data[img_id].width, filtered_data[img_id].height) <= self.max_resolution}
            Logger.log('Removed %d high resolution images' % (initial_len - len(filtered_data)))

        Logger.log('Final images amount is %d' % len(filtered_data))

        return filtered_data

    def _filter_dict_keys(self, dict, keys):
        return {k: dict[k] for k in dict if k in keys}

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
        data = self._filter_images(aerial_data.data)
        self._fix_exceeding_bounding_boxes(data)
        train_data_len = int((self.train_split_percent / float(100)) * len(data))
        train_data = random.sample(list(data), train_data_len)
        val_data = [val_data for val_data in data if val_data not in train_data]
        assert set(train_data) != set(val_data)
        train_data = self._filter_dict_keys(data, train_data)
        val_data = self._filter_dict_keys(data, val_data)
        trainval_data = {}
        trainval_data.update(train_data)
        trainval_data.update(val_data)
        self._assert_dirs()
        self._gen_train_file(train_data)
        self._gen_val_file(val_data)
        self._gen_trainval_file(trainval_data)
        self._gen_annotations(trainval_data)


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
                        default=True,
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
    pascal_voc = PascalVoc(args.data_path,
                           args.train_split_percent,
                           args.limit_resolution,
                           args.max_resolution)
    pascal_voc.convert(data)
