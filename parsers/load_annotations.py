import csv
import json
import argparse

def ComplexHandler(Obj):
    if hasattr(Obj, 'jsonable'):
        return Obj.jsonable()
    else:
        raise TypeError('Object of type %s with value of %s is not JSON serializable' % (type(Obj), repr(Obj)))

class AerialData:
    class InnerData:
        def __init__(self, image_id):
            self.image_id = image_id
            self.category = ''

    def __init__(self):
        self.data = {}

    def jsonable(self):
        return self.__dict__

    def load(self, train_details_file, train_tags_file):
        with open(train_details_file, 'rt') as train_details:
            print('Loading aerial data files...')
            reader = csv.reader(train_details, delimiter=',', quotechar='"')
            # skip first row
            next(reader)
            for row in reader:
                # read imageid and category
                image_id = row[0]
                cat = row[1]
                if image_id not in self.data:
                    self.data[image_id] = self.InnerData(image_id)
                self.data[image_id].category = cat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loads aerial annotations')
    parser.add_argument('--train_tags', dest='train_tags', type=str,
                        default='../Detecting And Classifying Objects In Aerial Imagery/Train/CSV/Train_tags.csv', help='train tags file path')
    parser.add_argument('--train_details', dest='train_details', type=str,
                        default='../Detecting And Classifying Objects In Aerial Imagery/Train/CSV/Train_imagery_details.csv',
                        help='train tags file path')
    args = parser.parse_args()
    print(args.train_details, args.train_tags)
    data = AerialData()
    data.load(args.train_details, args.train_tags)
    print(json.dumps(data.__dict__, default=ComplexHandler))