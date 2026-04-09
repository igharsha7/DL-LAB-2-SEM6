# Program 9: Regularized Deep Autoencoder

## Aim
To build a regularized deep autoencoder for image reconstruction and reduce overfitting using dropout and L2 regularization.

## Theory
An autoencoder is a neural network that learns to reconstruct input data through:
- Encoder: compresses input into a lower-dimensional latent representation.
- Decoder: reconstructs original input from latent space.

Deep autoencoders may overfit, so regularization is added:
- L2 regularization penalizes large weights.
- Dropout randomly disables neurons during training.

For reconstruction, Mean Squared Error (MSE) is commonly used:

$$MSE = \frac{1}{n}\sum_{i=1}^{n}(x_i-\hat{x}_i)^2$$

Lower MSE indicates better reconstruction quality.

## Code
```python
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.regularizers import l2

# Load Fashion-MNIST dataset
(x_train, _), (x_test, _) = fashion_mnist.load_data()

# Normalize and flatten images (28x28 -> 784)
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

input_dim = 784

# Build Regularized Deep Autoencoder
input_layer = Input(shape=(input_dim,))

# Encoder
x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(input_layer)
x = Dropout(0.3)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
encoded = Dense(64, activation='relu')(x)

# Decoder
x = Dense(256, activation='relu')(encoded)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
decoded = Dense(input_dim, activation='sigmoid')(x)

autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train model
history = autoencoder.fit(
    x_train, x_train,
    epochs=20,
    batch_size=128,
    validation_data=(x_test, x_test),
    verbose=1
)

# Reconstruct sample test images
decoded_imgs = autoencoder.predict(x_test[:5])

# Display original vs reconstructed images
plt.figure(figsize=(10, 4))
for i in range(5):
    # Original
    ax = plt.subplot(2, 5, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
    ax.set_title("Original")
    plt.axis('off')

    # Reconstructed
    ax = plt.subplot(2, 5, i + 6)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap='gray')
    ax.set_title("Reconstructed")
    plt.axis('off')

plt.tight_layout()
plt.show()
```

## Expected Output
```text
Epoch 1/20
...
Epoch 20/20
loss: 0.0xxx - val_loss: 0.0xxx
```

Also, a figure is displayed with:
- Top row: original Fashion-MNIST images
- Bottom row: reconstructed images from the regularized autoencoder

Note: Loss values and reconstructed quality vary slightly across runs.
