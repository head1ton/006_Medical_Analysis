import keras

binary_loss_object = keras.losses.BinaryCrossentropy(from_logits=False)

sparse_categorical_cross_entropy_object = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
