import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
import numpy as np

(x_train, y_train), _ = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255
x_train = np.expand_dims(x_train, -1)
y_train = tf.keras.utils.to_categorical(y_train, 10)

noise = 0.05 * np.random.normal(size=x_train.shape)
x_train = np.clip(x_train + noise, 0., 1.)

datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    validation_split=0.2
)
datagen.fit(x_train)

model = keras.Sequential()
model.add(layers.Flatten(input_shape=(28, 28, 1)))
model.add(layers.Dense(
    256, activation='relu',
    kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4)
))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(
    128, activation='relu',
    kernel_regularizer=regularizers.l2(1e-4)
))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    datagen.flow(x_train, y_train, batch_size=128, subset='training'),
    validation_data=datagen.flow(
        x_train, y_train, batch_size=128, subset='validation'
    ),
    epochs=50,
    callbacks=[early_stop]
)
