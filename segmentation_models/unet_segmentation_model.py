import segmentation_models as sm
import albumentations as A
import numpy as np
import cv2
import os
import keras
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os

import albumentations as A
import cv2
import keras
import numpy as np
import segmentation_models as sm
from keras._tf_keras.keras.callbacks import ModelCheckpoint, EarlyStopping, \
    ReduceLROnPlateau

sm.set_framework('tf.keras')
sm.framework()

# Config
IMAGE_SIZE = 512
BATCH_SIZE = 6 # 8
BACKBONE = 'efficientnetb0'
CLASSES = 1
ACTIVATION = 'sigmoid'
EPOCHS = 40
LEARNING_RATE = 1e-4

# Model
model = sm.Unet(
    backbone_name=BACKBONE,
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    classes=CLASSES,
    activation=ACTIVATION,
    encoder_weights='imagenet'
)

# Loss and Metrics
dice_loss = sm.losses.DiceLoss()
bce_loss = sm.losses.BinaryCELoss()
total_loss = dice_loss + bce_loss

metrics = [
    sm.metrics.IOUScore(threshold=0.5),
    sm.metrics.FScore(threshold=0.5),
    'accuracy'
]

model.compile(optimizer=keras.optimizers.Adam(LEARNING_RATE), loss=total_loss, metrics=metrics)


# DataLoader
# DataLoader
class Dataset(keras.utils.Sequence):
    def __init__(self, image_paths, mask_paths, augment=None,
        batch_size=BATCH_SIZE):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, idx):
        batch_x = self.image_paths[
                  idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.mask_paths[
                  idx * self.batch_size:(idx + 1) * self.batch_size]

        images, masks = [], []
        for img_path, mask_path in zip(batch_x, batch_y):
            img = cv2.imread(img_path)
            if img is None:
                raise ValueError(f"Image not found at path: {img_path}")
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise ValueError(f"Mask not found at path: {mask_path}")
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))
            mask = np.expand_dims(mask, axis=-1)
            mask = mask.astype(np.float32) / 255.0

            if self.augment:
                augmented = self.augment(image=img, mask=mask)
                img = augmented['image']
                mask = augmented['mask']

            images.append(img)
            masks.append(mask)

        # print(f"[DEBUG] Batch {idx}: {len(images)} images, {len(masks)} masks")

        return np.array(images), np.array(masks)

def get_train_arg():
    return A.Compose([
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomBrightnessContrast(),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15),
        A.Normalize()
    ])

def get_val_arg():
    return A.Compose([A.Normalize()])

def create_mask(pred_mask):
    pred_mask = np.where(pred_mask > 0.5, 1, 0)

    return pred_mask[0]

train_imgs = sorted([f"datasets/train/images/{f}" for f in os.listdir("datasets/train/images")])
train_masks = sorted([f"datasets/train/masks/{f}" for f in os.listdir("datasets/train/masks")])
val_imgs = sorted([f"datasets/val/images/{f}" for f in os.listdir("datasets/val/images")])
val_masks = sorted([f"datasets/val/masks/{f}" for f in os.listdir("datasets/val/masks")])

train_dataset = Dataset(train_imgs, train_masks, augment=get_train_arg())
val_dataset = Dataset(val_imgs, val_masks, augment=get_val_arg())

callbacks = [
    ModelCheckpoint("best_model.h5", save_best_only=True, monitor='val_loss', mode='min'),
    EarlyStopping(patience=10, monitor='val_loss', mode='min'),
    ReduceLROnPlateau(patience=1, monitor='val_loss', mode='min', factor=0.5)
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=callbacks
)

# predict & visualize
import matplotlib.pyplot as plt

def visualize(img, mask, pred):
    plt.figure(figsize=(16,8))
    plt.subplot(1,3,1); plt.imshow(keras.preprocessing.image.array_to_img(img)); plt.title("Image")
    plt.subplot(1,3,2); plt.imshow(keras.preprocessing.image.array_to_img(mask), cmap='gray'); plt.title("Ground Truth")
    plt.subplot(1,3,3); plt.imshow(keras.preprocessing.image.array_to_img(pred), cmap='gray'); plt.title("Prediction")
    plt.show()


img, mask = val_dataset[0]
pred = model.predict(img)

pred = np.where(pred > 0.5, 1, 0)

print(f"Image shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}, min: {mask.min()}, max: {mask.max()}")
print(f"pred shape: {pred.shape}, dtype: {pred.dtype}, min: {pred.min()}, max: {pred.max()}")

visualize(img[0], mask[0], pred[0])