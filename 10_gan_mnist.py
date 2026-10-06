import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

# Load and normalize MNIST
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=-1)

batch_size = 128
buffer_size = 60000
epochs = 20
noise_dim = 100

train_dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .shuffle(buffer_size)
    .batch(batch_size)
)

# Generator
def build_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 128, input_dim=noise_dim))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Reshape((7, 7, 128)))
    model.add(layers.Conv2DTranspose(
        64, (5,5), strides=2, padding="same"
    ))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Conv2DTranspose(
        1, (5,5), strides=2, padding="same",
        activation="tanh"
    ))
    return model

# Discriminator
def build_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(
        64, (5,5), strides=2, padding="same",
        input_shape=[28,28,1]
    ))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Conv2D(
        128, (5,5), strides=2, padding="same"
    ))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Flatten())
    model.add(layers.Dense(1))
    return model

generator = build_generator()
discriminator = build_discriminator()

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

def discriminator_loss(real_output, fake_output):
    real_loss = loss_fn(tf.ones_like(real_output), real_output)
    fake_loss = loss_fn(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return loss_fn(tf.ones_like(fake_output), fake_output)

@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gen_gradients = gen_tape.gradient(
        gen_loss, generator.trainable_variables
    )
    disc_gradients = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gen_gradients, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(disc_gradients, discriminator.trainable_variables)
    )

    return gen_loss, disc_loss

for epoch in range(epochs):
    for images in train_dataset:
        # Last batch can be smaller; use only full batches
        if images.shape[0] != batch_size:
            continue
        gen_loss, disc_loss = train_step(images)

    print(
        f"Epoch {epoch + 1}/{epochs} - "
        f"Generator Loss: {gen_loss:.4f}, "
        f"Discriminator Loss: {disc_loss:.4f}"
    )

# Generate sample images
noise = tf.random.normal([16, noise_dim])
generated_images = generator(noise, training=False)

plt.figure(figsize=(8, 8))

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.imshow(
        (generated_images[i] + 1) / 2,
        cmap='gray'
    )
    plt.axis('off')

plt.tight_layout()
plt.show()
