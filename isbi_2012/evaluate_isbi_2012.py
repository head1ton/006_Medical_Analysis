import os

import numpy as np
import skimage.io as io
from absl import app
from absl import flags
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

from model import UNET_ISBI_2012

flags.DEFINE_string('checkpoint_path', default='./isbi_2012/saved_model_isbi_2012/unet_model.h5', help='path to a directory to restore checkpoint file.')
flags.DEFINE_string('test_dir', default='./isbi_2012/isbi_2012_test_result', help='path to a directory to save test result.')
flags.DEFINE_integer('num_classes', default=1, help='number of classes to predict.')

FLAGS = flags.FLAGS

batch_size = 1
total_test_image_num = 30


def normalize_isbi_2012(input_images):
    input_images = input_images / 255.0

    return input_images


def make_test_generator(batch_size):
    image_gen = ImageDataGenerator()

    image_generator = image_gen.flow_from_directory(
        directory='./isbi_2012/isbi_2012_dataset/datasets',
        classes=['test_images'],
        class_mode=None,
        target_size=(512, 512),
        batch_size=batch_size,
        color_mode='grayscale',
        seed=1
    )

    for batch_images in image_generator:
        batch_images = normalize_isbi_2012(batch_images)

        yield batch_images


def create_mask(pred_mask):
    pred_mask = np.where(pred_mask > 0.5, 1, 0)

    return pred_mask[0]


def main():
    if not os.path.exists(FLAGS.checkpoint_path):
        print('checkpoint file is not exists!')
        exit()

    unet_model = UNET_ISBI_2012(FLAGS.num_classes)

    unet_model.load_weights(FLAGS.checkpoint_path)
    print(f'{FLAGS.checkpoint_path} checkpoint is restored!')

    test_generator = make_test_generator(batch_size)

    print('total test image : ', total_test_image_num)

    if not os.path.exists(os.path.join(os.getcwd(), FLAGS.test_dir)):
        os.mkdir(os.path.join(os.getcwd(), FLAGS.test_dir))

    for image_num, test_image in enumerate(test_generator):
        if image_num >= total_test_image_num:
            break
        pred_mask = unet_model.predict(test_image)

        output_image_path = os.path.join(os.getcwd(), FLAGS.test_dir, f'{image_num}_result.png')
        io.imsave(output_image_path, create_mask(pred_mask))
        print(output_image_path + ' saved!')


if __name__ == "__main__":
    app.run(main)