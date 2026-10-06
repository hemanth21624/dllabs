import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt
import numpy as np

# Load MNIST
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

# Add noise
x_train_noisy = np.clip(
    x_train + 0.3 * np.random.normal(0, 1, x_train.shape),
    0, 1
)
x_test_noisy = np.clip(
    x_test + 0.3 * np.random.normal(0, 1, x_test.shape),
    0, 1
)

# Create autoencoder
model = Sequential()

# Encoder
model.add(Conv2D(
    32, (3,3), activation='relu',
    padding='same', input_shape=(28,28,1)
))
model.add(MaxPooling2D((2,2), padding='same'))
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(MaxPooling2D((2,2), padding='same'))

# Decoder
model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(32, (3,3), activation='relu', padding='same'))
model.add(UpSampling2D((2,2)))
model.add(Conv2D(
    1, (3,3), activation='sigmoid', padding='same'
))

model.compile(optimizer='adam', loss='binary_crossentropy')

model.fit(
    x_train_noisy,
    x_train,
    epochs=5,
    batch_size=256,
    validation_data=(x_test_noisy, x_test)
)

decoded = model.predict(x_test_noisy, verbose=0)

# Show original, noisy and reconstructed images
plt.figure(figsize=(10, 6))

for i in range(10):
    plt.subplot(3, 10, i + 1)
    plt.imshow(x_test[i].squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(3, 10, i + 11)
    plt.imshow(x_test_noisy[i].squeeze(), cmap='gray')
    plt.axis('off')

    plt.subplot(3, 10, i + 21)
    plt.imshow(decoded[i].squeeze(), cmap='gray')
    plt.axis('off')

plt.show()
