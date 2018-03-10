import os
import sys
from PIL import Image

_max = 0
_min = 100000
first_bucket = []
second_bucket = []
third_bucket = []
fourth_bucket = []
fifth_bucket = []
total_ims = 0
for filename in os.listdir(sys.argv[1]):
	im = Image.open(os.path.join(sys.argv[1],filename))
	total_ims += 1
	width, height = im.size
	if max(im.size) > _max:
		_max = max(im.size)
		max_dim = im.size
	if min(im.size) < _min:
		_min = min(im.size)
		min_dim = im.size
	
	if max(im.size) <= 1000:
		first_bucket.append(im.size)
	elif max(im.size) <= 2000:
                second_bucket.append(im.size)
	elif max(im.size) <= 3000:
                third_bucket.append(im.size)
	elif max(im.size) <= 4000:
                fourth_bucket.append(im.size)
	elif max(im.size) <= 5000:
                fifth_bucket.append(im.size)


print('max dims: ' + str(max_dim))
print('min dims: ' + str(min_dim))
print('# of images: ' + str(total_ims))
print('# of images in [0, 1000]: ' + str(len(first_bucket)))
print('# of images in [1000, 2000]: ' + str(len(second_bucket)))
print('# of images in [2000, 3000]: ' + str(len(third_bucket)))
print('# of images in [3000, 4000]: ' + str(len(fourth_bucket)))
print('# of images in [4000, 5000]: ' + str(len(fifth_bucket)))

