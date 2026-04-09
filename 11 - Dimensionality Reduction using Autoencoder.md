# Program 11: Dimensionality Reduction using Autoencoder

## Aim
To demonstrate dimensionality reduction using an autoencoder by compressing image vectors into a lower-dimensional latent representation.

## Theory
Dimensionality reduction transforms high-dimensional data into fewer features while preserving important information.

In an autoencoder:
- Encoder maps input $x$ to latent vector $z$.
- Decoder reconstructs $\hat{x}$ from $z$.

If input dimension is large (for example 784 for 28x28 images) and latent dimension is small (for example 64), then $z$ is the reduced representation.

Reconstruction is trained using mean squared error:

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{x}_i)^2$$

After training, the encoder output is used as reduced data.

## Code
```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.datasets import fashion_mnist

# Load Fashion-MNIST
(x_train, _), (x_test, _) = fashion_mnist.load_data()
X = np.concatenate([x_train, x_test], axis=0)

# Preprocess: flatten and normalize
X = X.reshape((len(X), -1)).astype("float32") / 255.0
input_dim = 784
encoding_dim = 64

# Build Autoencoder
input_layer = Input(shape=(input_dim,))

# Encoder
encoded = Dense(256, activation='relu')(input_layer)
encoded = Dense(128, activation='relu')(encoded)
latent = Dense(encoding_dim, activation='relu')(encoded)

# Decoder
decoded = Dense(128, activation='relu')(latent)
decoded = Dense(256, activation='relu')(decoded)
output_layer = Dense(input_dim, activation='sigmoid')(decoded)

autoencoder = Model(input_layer, output_layer)
encoder = Model(input_layer, latent)

autoencoder.compile(optimizer='adam', loss='mse')

# Train
autoencoder.fit(
    X, X,
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    shuffle=True,
    verbose=1
)

# Dimensionality reduction
reduced_data = encoder.predict(X)
print("Original shape:", X.shape)
print("Reduced shape:", reduced_data.shape)

# Reconstruction check
decoded_imgs = autoencoder.predict(X[:5])

plt.figure(figsize=(10, 4))
for i in range(5):
    # Original
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(X[i].reshape(28, 28), cmap='gray')
    ax.set_title("Original")
    ax.axis('off')

    # Reconstructed
    ax = plt.subplot(2, 5, i + 6)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    ax.set_title("Reconstructed")
    ax.axis('off')

plt.tight_layout()
plt.show()
```

## Expected Output
```text
Epoch 1/10
...
Epoch 10/10
Original shape: (70000, 784)
Reduced shape: (70000, 64)
```

Also, a figure is displayed with original images and their reconstructed versions.

Note: Exact loss values vary slightly in each run.
