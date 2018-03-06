import csv
import json
import argparse


class AerialData:
    class BoundingBox:
        def __init__(self, pt1, pt2, pt3, pt4):
            self.pt1 = pt1
            self.pt2 = pt2
            self.pt3 = pt3
            self.pt4 = pt4

    class LargeVehicleFeatures:
        def __init__(self, subclass,
                     open_cargo_area,
                     vents,
                     air_conditioner,
                     wrecked,
                     enclosed_box,
                     enclosed_cab,
                     ladder,
                     flatbed,
                     soft_shell_box,
                     harnessed_to_cart,
                     color
                     ):
            self.subclass = subclass
            self.open_cargo_area = open_cargo_area
            self.vents = vents
            self.air_conditioner = air_conditioner
            self.wrecked = wrecked
            self.enclosed_box = enclosed_box
            self.enclosed_cab = enclosed_cab
            self.ladder = ladder
            self.flatbed = flatbed
            self.soft_shell_box = soft_shell_box
            self.harnessed_to_cart = harnessed_to_cart
            self.color = color

    class SmallVehicleFeatures:
        def __init__(self, subclass,
                     sunroof,
                     taxi,
                     luggage_carrier,
                     open_cargo_area,
                     enclosed_cab,
                     wrecked,
                     spare_wheel,
                     color):
            self.subclass = subclass
            self.sunroof = sunroof
            self.taxi = taxi
            self.luggage_carrier = luggage_carrier
            self.open_cargo_area = open_cargo_area
            self.enclosed_cab = enclosed_cab
            self.wrecked = wrecked
            self.spare_wheel = spare_wheel
            self.color = color

    class InnerData:
        def __init__(self, image_id):
            self.image_id = image_id
            self.category = None
            self.bounding_box = {}
            self.label = None
            self.sublabel = None

    def __init__(self):
        self.data = {}

    def jsonable(self):
        return self.__dict__

    def create_box(self, row):
        return self.BoundingBox((row[1], row[2]),
                                (row[3], row[4]),
                                (row[5], row[6]),
                                (row[7], row[8]))

    def load(self, train_details_file, train_tags_file):
        print('Loading aerial data files...')
        with open(train_details_file, 'rt') as train_details:
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
        with open(train_tags_file, 'rt') as train_tags:
            reader = csv.reader(train_tags, delimiter=',', quotechar='"')
            next(reader)
            for row in reader:
                image_id = row[0]
                if image_id not in self.data:
                    raise RuntimeError('Image id %s in train tags was not present in train details' % image_id)
                label = row[9]
                inner_data = self.data[image_id]
                inner_data.label = label
                subclass, sunroof, taxi, luggage_carrier, \
                open_cargo_area, enclosed_cab, spare_wheel, \
                wrecked, flatbed, ladder, \
                enclosed_box, soft_shell_box, harnessed_to_cart, \
                vents, air_conditioner, color = row[10:len(row)]
                if label == 'large vehicle':
                    inner_data.bounding_box = self.create_box(row)
                    inner_data.sublabel = self.LargeVehicleFeatures(subclass,
                                                                    open_cargo_area,
                                                                    vents,
                                                                    air_conditioner,
                                                                    wrecked,
                                                                    enclosed_box,
                                                                    enclosed_cab,
                                                                    ladder,
                                                                    flatbed,
                                                                    soft_shell_box,
                                                                    harnessed_to_cart,
                                                                    color)
                elif label == 'small vehicle':
                    inner_data.bounding_box = self.create_box(row)
                    inner_data.sublabel = self.SmallVehicleFeatures(subclass,
                                                                    sunroof,
                                                                    taxi,
                                                                    luggage_carrier,
                                                                    open_cargo_area,
                                                                    enclosed_cab,
                                                                    wrecked,
                                                                    spare_wheel,
                                                                    color)
                elif label == 'solar panel':
                    inner_data.bounding_box = self.create_box(row)
                elif label == 'utility pole':
                    inner_data.bounding_box = self.BoundingBox((row[1], row[2]),
                                                               (row[1], row[2]),
                                                               (row[1], row[2]),
                                                               (row[1], row[2]))
                else:
                    raise RuntimeError('Unrecognized class %s' % label)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Loads aerial annotations')
    parser.add_argument('--train_tags', dest='train_tags', type=str,
                        default='../Detecting And Classifying Objects In Aerial Imagery/Train/CSV/Train_tags.csv',
                        help='train tags file path')
    parser.add_argument('--train_details', dest='train_details', type=str,
                        default='../Detecting And Classifying Objects In Aerial Imagery/Train/CSV/Train_imagery_details.csv',
                        help='train tags file path')
    args = parser.parse_args()
    print(args.train_details, args.train_tags)
    data = AerialData()
    data.load(args.train_details, args.train_tags)
