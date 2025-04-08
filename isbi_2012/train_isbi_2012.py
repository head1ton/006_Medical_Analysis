import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from absl import app
from absl import flags
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

from loss import binary_loss_object
from model import UNET_ISBI_2012

tf.random.set_seed(4000)

flags.DEFINE_string('checkpoint_path', default='isbi_2012/isbi_2012_dataset/saved_model_isbi_2012/unet_model.h5', help='Path to save the model checkpoint')
flags.DEFINE_string('tensorboard_log_path', default='isbi_2012/isbi_2012_dataset/tensorboard_log_isbi_2012', help='Path to save the tensorboard logs')
flags.DEFINE_integer('num_epochs', default=5, help='Number of epochs to train the model')
flags.DEFINE_integer('steps_per_epoch', default=2000, help='Number of steps per epoch')
flags.DEFINE_integer('num_classes', default=1, help='Number of classes in the dataset')

FLAGS = flags.FLAGS

batch_size = 2
learning_rate = 0.0001


def normalize_isbi_2012(image_gen, mask_gen):
    input_images = image_gen / 255
    mask_labels = mask_gen / 255

    mask_labels[mask_labels > 0.5] = 1
    mask_labels[mask_labels <= 0.5] = 0

    return input_images, mask_labels


def make_train_generator(batch_size, aug_dict):
    """
    Create a training data generator with data augmentation.

    Args:
        batch_size (int): Size of the batches of data.
        aug_dict (dict): Dictionary containing augmentation parameters.

    Returns:
        train_generator: A generator that yields augmented training data.
    """
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)

    image_generator = image_datagen.flow_from_directory(
        directory='./isbi_2012/isbi_2012_dataset/datasets',
        classes=['train_images'],
        class_mode=None,
        target_size=(512, 512),
        batch_size=batch_size,
        color_mode='grayscale',
        seed=1
    )

    mask_generator = mask_datagen.flow_from_directory(
        directory='./isbi_2012/isbi_2012_dataset/datasets',
        classes=['train_labels'],
        class_mode=None,
        target_size=(512, 512),
        batch_size=batch_size,
        color_mode='grayscale',
        seed=1
    )

    # Assuming you have a directory structure for your dataset
    train_generator = zip(image_generator, mask_generator)

    for (image_gen, mask_gen) in train_generator:
        batch_images, batch_labels = normalize_isbi_2012(image_gen, mask_gen)

        yield batch_images, batch_labels

def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    pred_mask = np.where(pred_mask > 0.5, 1, 0)

    return pred_mask[0]


def show_predictions(model, sample_image, sample_mask):
    display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])


def display_and_save(display_list, epoch):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')

    plt.savefig(f'Epoch {epoch}.jpg')

def save_predictions(epoch, model, sample_image, sample_mask):
    display_and_save([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))], epoch)

class CustomCallback(keras.callbacks.Callback):
    def __init__(self, unet_model, sample_image, sample_mask):
        super(CustomCallback, self).__init__()
        self.unet_model = unet_model
        self.sample_image = sample_image
        self.sample_mask = sample_mask

    def on_epoch_end(self, epoch, logs=None):
        save_predictions(epoch + 1, self.unet_model, self.sample_image, self.sample_mask)
        print(f'\nEpoch {epoch + 1} ended. Predictions saved!')


def main(_):

    # Argumentation
    aug_dict = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

    train_generator = make_train_generator(batch_size, aug_dict)

    # sample_image = None
    # sample_mask = None

    for i, batch_data in enumerate(train_generator):
        # if i >= 2:
        #     break
        batch_image, batch_mask = batch_data[0], batch_data[1]
        sample_image, sample_mask = batch_image[0], batch_mask[0]

        display([sample_image, sample_mask])

        unet_model = UNET_ISBI_2012(FLAGS.num_classes)

        show_predictions(unet_model, sample_image, sample_mask)

        optimizer = keras.optimizers.Adam(learning_rate)

        if not os.path.exists(FLAGS.checkpoint_path.split('/')[0]):
            os.mkdir(FLAGS.checkpoint_path.split('/')[0])

        if os.path.isfile(FLAGS.checkpoint_path):
            unet_model.load_weights(FLAGS.checkpoint_path)
            print(f'{FLAGS.checkpoint_path} checkpoint is restored!')

        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(FLAGS.checkpoint_path, monitor='loss', verbose=1, save_best_only=True)
        tensorboard_callback = keras.callbacks.TensorBoard(log_dir=FLAGS.tensorboard_log_path)
        custom_callback = CustomCallback(unet_model, sample_image, sample_mask)

        unet_model.compile(optimizer=optimizer, loss=binary_loss_object, metrics=['accuracy'])

        unet_model.fit(train_generator,
                       steps_per_epoch=FLAGS.steps_per_epoch,
                       epochs=FLAGS.num_epochs,
                       callbacks=[model_checkpoint_callback, tensorboard_callback, custom_callback]
                       )


if __name__ == "__main__":
    app.run(main)

