# Program 8: Image Generation using GANs

## Aim
To implement image generation using a Generative Adversarial Network (GAN) and produce synthetic grayscale images.

## Theory
A GAN consists of two neural networks trained together:
- Generator: Creates fake images from random noise.
- Discriminator: Classifies images as real or fake.

Training is adversarial:
1. Train discriminator on real and generated images.
2. Train generator to fool discriminator.

Over epochs, generator learns the data distribution and produces realistic images.

Binary cross-entropy loss is commonly used:

$$L_D = -\left[\log D(x) + \log(1-D(G(z)))\right]$$

$$L_G = -\log D(G(z))$$

Where:
- $x$ is a real image.
- $z$ is random noise.
- $G(z)$ is generated image.
- $D(\cdot)$ is discriminator output probability.

## Code
```python
import tensorflow as tf
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt

# Load MNIST and normalize to [-1, 1]
(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype("float32") - 127.5) / 127.5
x_train = x_train.reshape(-1, 28 * 28)

dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(128)
noise_dim = 100

# Generator
generator = Sequential([
    layers.Dense(128, input_shape=(noise_dim,)),
    layers.LeakyReLU(),
    layers.Dense(256),
    layers.LeakyReLU(),
    layers.Dense(28 * 28, activation="tanh")
])

# Discriminator
discriminator = Sequential([
    layers.Dense(256, input_shape=(28 * 28,)),
    layers.LeakyReLU(),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

bce = tf.keras.losses.BinaryCrossentropy()
g_opt = tf.keras.optimizers.Adam(1e-4)
d_opt = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(real_images):
    batch_size = tf.shape(real_images)[0]
    noise = tf.random.normal([batch_size, noise_dim])

    # Train Discriminator
    with tf.GradientTape() as d_tape:
        fake_images = generator(noise, training=True)
        real_out = discriminator(real_images, training=True)
        fake_out = discriminator(fake_images, training=True)

        d_loss = bce(tf.ones_like(real_out), real_out) + bce(tf.zeros_like(fake_out), fake_out)

    d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
    d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    # Train Generator
    noise = tf.random.normal([batch_size, noise_dim])
    with tf.GradientTape() as g_tape:
        fake_images = generator(noise, training=True)
        fake_out = discriminator(fake_images, training=True)
        g_loss = bce(tf.ones_like(fake_out), fake_out)

    g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

def show_generated_images(epoch, n=16):
    noise = tf.random.normal([n, noise_dim])
    images = generator(noise, training=False)
    images = (images + 1) / 2.0

    plt.figure(figsize=(4, 4))
    for i in range(n):
        plt.subplot(4, 4, i + 1)
        plt.imshow(images[i].numpy().reshape(28, 28), cmap="gray")
        plt.axis("off")
    plt.suptitle(f"Epoch {epoch}")
    plt.show()

# Training loop
epochs = 10
for epoch in range(epochs):
    for batch in dataset:
        train_step(batch)
    print(f"Epoch {epoch + 1} completed")

# Display generated images
show_generated_images(epochs)
```

## Expected Output
```text
Epoch 1 completed
Epoch 2 completed
...
Epoch 10 completed
```

And a 4x4 grid of generated grayscale digit-like images is displayed.

Note: Generated image quality improves with more epochs (for example 30 to 100 epochs).
