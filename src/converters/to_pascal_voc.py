import argparse
import random
import os
from converters.load_annotations import AerialData


class PascalVoc:
    def __init__(self, train_split_percent):
        self.train_split_percent = train_split_percent
        self.root_dirname = 'VOCdevkit2007/VOC2007'
        self.image_ids_dirname = 'ImageSets/Main'
        self.train_fname = 'train.txt'
        self.val_fname = 'val.txt'
        self.trainval_fname = 'trainval.txt'

    def _gen_train_file(self, train_data):
        fpath = os.path.join(self.root_dirname, self.image_ids_dirname, self.train_fname)
        content = ['%s\n' % key for key in train_data]
        with open(fpath, 'w') as train_file:
            train_file.write(content)
            print('Generated train file with ids')

    def _gen_val_file(self, val_data):
        fpath = os.path.join(self.root_dirname, self.image_ids_dirname, self.val_fname)
        content = ['%s\n' % key for key in val_data]
        with open(fpath, 'w') as val_file:
            val_file.write(content)
            print('Generated val file with ids')

    def _gen_trainval_file(self, trainval_data):
        fpath = os.path.join(self.root_dirname, self.image_ids_dirname, self.trainval_fname)
        content = ['%s\n' % key for key in trainval_data]
        with open(fpath, 'w') as trainval_file:
            trainval_file.write(content)
            print('Generated val file with ids')

    def convert(self, aerial_data):
        train_data_len = (self.train_split_percent / float(100)) * len(aerial_data.data)
        train_data = random.sample(aerial_data.data, train_data_len)
        val_data = [data for data in aerial_data.data if data not in train_data]
        assert set(train_data) != set(val_data)
        self._gen_train_file(train_data)
        self._gen_val_file(val_data)
        self._gen_trainval_file(aerial_data.data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loads aerial annotations')
    parser.add_argument('--train_tags', dest='train_tags', type=str,
                        default='../Detecting And Classifying Objects In Aerial Imagery/Train/CSV/Train_tags.csv',
                        help='train tags file path')
    parser.add_argument('--train_details', dest='train_details', type=str,
                        default='../Detecting And Classifying Objects In Aerial Imagery/Train/CSV/Train_imagery_details.csv',
                        help='train tags file path')
    parser.add_argument('--train_split_percent', dest='train_split_percent', type=str,
                        default='../Detecting And Classifying Objects In Aerial Imagery/Train/CSV/Train_imagery_details.csv',
                        help='90', choices=range(1, 100))

    args = parser.parse_args()
    print(args.train_details, args.train_tags)
    data = AerialData()
    data.load(args.train_details, args.train_tags)
    pascal_voc = PascalVoc(args.train_split_percent)
    pascal_voc.convert(data)