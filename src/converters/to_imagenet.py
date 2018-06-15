import argparse
import random
import os
from shutil import copyfile
import _init_paths
from load_annotations import AerialData
from utils.logger import Logger
from PIL import Image


class ImageNet:
    def __init__(self, data_dir_path, train_split_percent, limit_resolution, max_resolution, images_dir, task):
        self.data_dir_path = data_dir_path
        self.train_split_percent = train_split_percent
        self.limit_resolution = limit_resolution
        self.max_resolution = max_resolution
        self.images_dir = images_dir
        self.task = task
        self.task_root_dir = task
        self.root_dirname = 'VOCdevkit2007'
        self.dataset_dirname = 'VOC2007'
        self.image_ids_dirname = 'ImageSets/Main'
        self.images_dirname = 'JPEGImages'
        self.annotations_dirname = 'Annotations'
        self.train_dirname = 'train'
        self.val_dirname = 'val'
        self.test_dirname = 'test'
        self.images_depth = 3

    def crop_image(self, fpath, bbox):
        img = Image.open(fpath)
        return img.crop(bbox), img.format

    def create_image(self, fpath, image, type):
        if type.lower() in ['jpg', 'jpeg']:
            image.save(fpath, 'jpeg', quality=95)
        elif type.lower() in ['tiff', 'tif']:
            image = image.convert('RGB')
            image.save(fpath, 'jpeg', quality=95)
        else:
            raise Exception('Unrecognized file format %s' % type)

    def _gen_dataset_file(self, data, data_dir):
        Logger.log('Beginning generating images to %s' % data_dir)
        images_root = os.path.join(self.data_dir_path, self.task_root_dir)
        for (idx, img_id) in enumerate(data, 1):
            if idx % 1000 == 0:
                Logger.log('Processed %d images...' % idx)
            img = data[img_id]
            img_fpath = img['fpath']
            cropped_obj, im_type = self.crop_image(img_fpath, img['bbox'])
            obj_path = os.path.join(images_root, data_dir, img[self.task].replace('/', '-'), img_id + '.jpg')
            self.create_image(obj_path, cropped_obj, im_type)
        Logger.log('Generated %s images' % data_dir)

    def _create_dirpath_if_not_exist(self, dirpath):
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)

    def _rm_dir(self, dir_path):
        for f in os.listdir(dir_path):
            file_path = os.path.join(dir_path, f)
            if os.path.isfile(file_path):
                os.unlink(file_path)

    def _assert_dirs(self, labels):
        images_root = os.path.join(self.data_dir_path, self.task_root_dir)
        self._create_dirpath_if_not_exist(images_root)
        self._rm_dir(images_root)
        for label in labels:
            label = label.replace('/', '-')
            train_label_dir_path = os.path.join(images_root, self.train_dirname, label)
            val_label_dir_path = os.path.join(images_root, self.val_dirname, label)
            test_label_dir_path = os.path.join(images_root, self.test_dirname, label)
            self._create_dirpath_if_not_exist(train_label_dir_path)
            self._create_dirpath_if_not_exist(val_label_dir_path)
            self._create_dirpath_if_not_exist(test_label_dir_path)

    def _obb_to_pascal_hbb(self, obb):
        xmin = min([float(x) for x, _ in obb])
        ymin = min([float(y) for _, y in obb])
        xmax = max([float(x) for x, _ in obb])
        ymax = max([float(y) for _, y in obb])
        return float(xmin), float(ymin), float(xmax), float(ymax)

    '''
    Remove:
    1. Untagged and No objects images
    2. Leave only Small and Large vehicle classes
    '''

    def _filter_images(self, data):
        initial_len = len(data)
        filtered_data = data
        Logger.log('Data contains %d images' % initial_len)

        # Remove untagged and no object images
        filtered_data = {img_id: filtered_data[img_id] for img_id in filtered_data if
                         filtered_data[img_id].category.lower() not in ['untagged', 'no objects']}
        Logger.log('Removed %d untagged and no object images' % (initial_len - len(filtered_data)))
        initial_len = len(filtered_data)

        # Remove high resolution images
        if self.limit_resolution:
            filtered_data = {img_id: filtered_data[img_id] for img_id in filtered_data if
                             max(filtered_data[img_id].width, filtered_data[img_id].height) <= self.max_resolution}
            Logger.log('Removed %d high resolution images' % (initial_len - len(filtered_data)))
            initial_len = len(filtered_data)

        filtered_data = self._filter_to_images_with_finegrained_classes(filtered_data)
        Logger.log('Final object images amount is %d' % len(filtered_data))
        return filtered_data

    def _filter_to_images_with_finegrained_classes(self, data):
        Logger.log('Converting images to object images...')
        results = {}
        for image_id in data:
            image = data[image_id]
            fpath = os.path.join(self.images_dir, image.fname)
            for (idx, obj) in enumerate(image.objects):
                if obj.label.lower() in ['large vehicle', 'small vehicle']:
                    label = getattr(obj.sublabel, self.task)
                    if not label or label == '':
                        Logger.log('Object in index #%d in image %s has no %s' % (idx + 1, image_id, self.task))
                        continue
                    new_obj = {
                        'fpath': fpath,
                        'height': image.height,
                        'width': image.width,
                        'bbox': obj.bounding_box.get_points(),
                        self.task: label
                    }
                    results[image_id + '_' + str(idx + 1)] = new_obj

        return results

    def _get_untagged_images(self, data):
        # Pick only untagged images
        filtered_data = {img_id: data[img_id] for img_id in data if
                         data[img_id].category == 'Untagged'}

        filtered_data = self._filter_to_images_with_finegrained_classes(filtered_data)
        Logger.log('Test object images amount is %d' % len(filtered_data))
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
        for id in data:
            obj_img = data[id]
            height = obj_img['height']
            width = obj_img['width']
            p1, p2, p3, p4 = obj_img['bbox']
            p1 = self._fix_single_point(id, p1, width, height)
            p2 = self._fix_single_point(id, p2, width, height)
            p3 = self._fix_single_point(id, p3, width, height)
            p4 = self._fix_single_point(id, p4, width, height)
            xmin, ymin, xmax, ymax = self._obb_to_pascal_hbb((p1, p2, p3, p4))
            assert xmin <= xmax < width and ymin <= ymax < height
            obj_img['bbox'] = xmin, ymin, xmax, ymax

    def convert(self, aerial_data):
        trainval_data = self._filter_images(aerial_data.data)
        test_data = self._get_untagged_images(aerial_data.data)
        self._fix_exceeding_bounding_boxes(trainval_data)
        train_data_len = int((self.train_split_percent / float(100)) * len(trainval_data))
        train_data = random.sample(list(trainval_data), train_data_len)
        val_data = [val_data for val_data in trainval_data if val_data not in train_data]
        assert set(train_data) != set(val_data) != set(test_data)
        train_data = self._filter_dict_keys(trainval_data, train_data)
        val_data = self._filter_dict_keys(trainval_data, val_data)
        trainval_data = {}
        trainval_data.update(train_data)
        trainval_data.update(val_data)
        labels = set([trainval_data[img_id][self.task] for img_id in trainval_data])
        self._assert_dirs(labels)
        self._gen_dataset_file(train_data, self.train_dirname)
        self._gen_dataset_file(val_data, self.val_dirname)


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
                        default='../resnet-50/data',
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
    parser.add_argument('--task', dest='task', type=str,
                        default='color',
                        help='the type of task we try to learn')

    args = parser.parse_args()
    if args.load_cached_aerial_data and AerialData.is_cached(args.data_path):
        data = AerialData.from_cache(args.data_path)
    else:
        data = AerialData(args.data_path)
        data.load(args.train_details, args.train_tags, args.images_dir)
    imagenet = ImageNet(args.data_path,
                        args.train_split_percent,
                        args.limit_resolution,
                        args.max_resolution,
                        args.images_dir,
                        args.task)
    imagenet.convert(data)
