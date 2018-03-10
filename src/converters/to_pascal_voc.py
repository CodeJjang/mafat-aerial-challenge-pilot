import argparse
import random
import os
import xml.etree.cElementTree as ET
from converters.load_annotations import AerialData
from utils.logger import Logger

class PascalVoc:
    def __init__(self, data_dir_path, train_split_percent):
        self.data_dir_path = data_dir_path
        self.train_split_percent = train_split_percent
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
        fpath = os.path.join(self.data_dir_path, self.root_dirname, self.dataset_dirname, self.image_ids_dirname, self.train_fname)
        content = '\n'.join(train_data)
        with open(fpath, 'w') as train_file:
            train_file.write(content)
            Logger.log('Generated train file with ids')

    def _gen_val_file(self, val_data):
        fpath = os.path.join(self.data_dir_path, self.root_dirname, self.dataset_dirname, self.image_ids_dirname, self.val_fname)
        content = '\n'.join(val_data)
        with open(fpath, 'w') as val_file:
            val_file.write(content)
            Logger.log('Generated val file with ids')

    def _gen_trainval_file(self, trainval_data):
        fpath = os.path.join(self.data_dir_path, self.root_dirname, self.dataset_dirname, self.image_ids_dirname, self.trainval_fname)
        content = '\n'.join(trainval_data)
        with open(fpath, 'w') as trainval_file:
            trainval_file.write(content)
            Logger.log('Generated val file with ids')

    def _gen_annotations(self, data):
        dirpath = os.path.join(self.data_dir_path, self.root_dirname, self.dataset_dirname, self.annotations_dirname)
        for image_id in data:
            image = data[image_id]
            if image.category == 'Untagged' or image.category == 'No objects':
                continue
            fpath = os.path.join(dirpath, image_id + '.xml')

            root = ET.Element("annotations")
            ET.SubElement(root, "folder").text = self.dataset_dirname
            ET.SubElement(root, "filename").text = image.fname

            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(image.width)
            ET.SubElement(size, "height").text = str(image.height)
            ET.SubElement(size, "depth").text = str(self.images_depth)

            ET.SubElement(root, "segmented").text = 0

            object = ET.SubElement(root, "object")
            ET.SubElement(object, "name").text = image.label
            ET.SubElement(object, "pose").text = 'Unspecified'
            ET.SubElement(object, "truncated").text = str(0)
            ET.SubElement(object, "difficult").text = str(0)
            bb = ET.SubElement(object, "bndbox")
            ET.SubElement(bb, 'xmin').text, \
            ET.SubElement(bb, 'ymin').text, \
            ET.SubElement(bb, 'xmax').text, \
            ET.SubElement(bb, 'ymax').text = self._obb_to_pascal_hbb(image.bounding_box.get_points())

            tree = ET.ElementTree(root)
            tree.write(fpath)
        Logger.log('Generated annotations')

    def _create_dirpath_if_not_exist(self, dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    def _assert_dirs(self):
        image_ids_dirs_path = os.path.join(self.data_dir_path, self.root_dirname, self.dataset_dirname, self.image_ids_dirname)
        images_dir_path = os.path.join(self.data_dir_path, self.root_dirname, self.dataset_dirname, self.images_dirname)
        annotations_dir_path = os.path.join(self.data_dir_path, self.root_dirname, self.dataset_dirname, self.annotations_dirname)
        self._create_dirpath_if_not_exist(image_ids_dirs_path)
        self._create_dirpath_if_not_exist(images_dir_path)
        self._create_dirpath_if_not_exist(annotations_dir_path)

    def _obb_to_pascal_hbb(self, obb):
        xmin = min([x for x, _ in obb])
        ymin = min([y for _, y in obb])
        xmax = max([x for x, _ in obb])
        ymax = max([y for _, y in obb])
        return str(xmin), str(ymin), str(xmax), str(ymax)

    def convert(self, aerial_data):
        train_data_len = int((self.train_split_percent / float(100)) * len(aerial_data.data))
        train_data = random.sample(list(aerial_data.data), train_data_len)
        val_data = [data for data in aerial_data.data if data not in train_data]
        assert set(train_data) != set(val_data)
        self._assert_dirs()
        self._gen_train_file(train_data)
        self._gen_val_file(val_data)
        self._gen_trainval_file(aerial_data.data)
        self._gen_annotations(aerial_data.data)


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

    args = parser.parse_args()
    Logger.log(args.train_details, args.train_tags)
    if args.load_cached_aerial_data and AerialData.is_cached(args.data_path):
        data = AerialData.from_cache(args.data_path)
    else:
        data = AerialData(args.data_path)
        data.load(args.train_details, args.train_tags, args.images_dir)
    pascal_voc = PascalVoc(args.data_path, args.train_split_percent)
    pascal_voc.convert(data)
