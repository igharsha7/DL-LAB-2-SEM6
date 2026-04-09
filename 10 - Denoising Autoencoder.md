# Program 10: Denoising Autoencoder

## Aim
To build an autoencoder that removes noise from corrupted images and reconstructs cleaner versions.

## Theory
A denoising autoencoder is trained to map noisy inputs to clean targets.

Main idea:
1. Add random noise to original images.
2. Feed noisy images to the autoencoder.
3. Train model to reconstruct the original clean images.

Architecture:
- Encoder compresses image features.
- Decoder reconstructs denoised image.

Loss used for reconstruction:

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{x}_i)^2$$

Where:
- $x_i$ is original pixel value.
- $\hat{x}_i$ is reconstructed pixel value.

## Code
```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt

# 1. Load Fashion-MNIST
(x_train, _), (x_test, _) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension
x_train = x_train[..., None]
x_test = x_test[..., None]

# 2. Add noise
noise_factor = 0.5
x_train_noisy = x_train + noise_factor * np.random.normal(size=x_train.shape)
x_test_noisy = x_test + noise_factor * np.random.normal(size=x_test.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
x_test_noisy = np.clip(x_test_noisy, 0., 1.)

# 3. Build denoising autoencoder
input_img = layers.Input(shape=(28, 28, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# 4. Train
history = autoencoder.fit(
    x_train_noisy, x_train,
    epochs=10,
    batch_size=64,
    shuffle=True,
    validation_data=(x_test_noisy, x_test),
    verbose=1
)

# 5. Predict denoised outputs
decoded_imgs = autoencoder.predict(x_test_noisy[:5])

# 6. Display noisy, original, reconstructed
plt.figure(figsize=(12, 6))
for i in range(5):
    # Noisy
    ax = plt.subplot(3, 5, i + 1)
    plt.imshow(x_test_noisy[i].reshape(28, 28), cmap='gray')
    ax.set_title("Noisy")
    ax.axis('off')

    # Original
    ax = plt.subplot(3, 5, i + 6)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.set_title("Original")
    ax.axis('off')

    # Reconstructed
    ax = plt.subplot(3, 5, i + 11)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    ax.set_title("Denoised")
    ax.axis('off')

plt.tight_layout()
plt.show()
```

## Expected Output
```text
Epoch 1/10
...
Epoch 10/10
loss: 0.0xxx - val_loss: 0.0xxx
```

Also, a visualization appears with 3 rows:
- Row 1: noisy input images
- Row 2: original clean images
- Row 3: denoised reconstructed images

Note: Exact loss values and denoising quality vary slightly in each run.
