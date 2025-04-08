import os

import skimage.io as io
import tifffile as tiff

train_image_path = os.path.join('datasets', 'train_images')
train_label_path = os.path.join('datasets', 'train_labels')
test_image_path = os.path.join('datasets', 'test_images')

train_images = tiff.imread(os.path.join('raw_data', 'train-volume.tif'))
train_labels = tiff.imread(os.path.join('raw_data', 'train-labels.tif'))
test_images = tiff.imread(os.path.join('raw_data', 'test-volume.tif'))

print(train_images.shape)
print(train_labels.shape)
print(test_images.shape)

if not os.path.exists(train_image_path):
    os.mkdir(train_image_path)
    os.mkdir(train_label_path)
    os.mkdir(test_image_path)

for index, images in enumerate(zip(train_images, train_labels, test_images)):
    tr_image, tr_label, tt_image = images

    io.imsave(os.path.join(train_image_path, f"{index}.png"), tr_image)
    io.imsave(os.path.join(train_label_path, f"{index}.png"), tr_label)
    io.imsave(os.path.join(test_image_path, f"{index}.png"), tt_image)

print("Isbi 2012 dataset images and labels have been saved to the respective folders.")

