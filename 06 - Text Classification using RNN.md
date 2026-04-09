# Program 6: Text Classification using RNN

## Aim
To implement text classification using a Recurrent Neural Network (RNN) model and classify text sentiment as positive or negative.

## Theory
Text classification assigns predefined labels to text data. In RNN-based text classification, tokenized text is converted into sequences and passed through an embedding layer and a recurrent layer (SimpleRNN/LSTM). The recurrent layer captures sequence context, and a dense output layer predicts class probability.

Pipeline:
1. Text preprocessing (tokenization + padding)
2. Label encoding
3. RNN model training
4. Prediction on new sentences

For binary classification, sigmoid output is used:

$$\hat{y}=\sigma(z)$$

If $\hat{y} > 0.5$, class is predicted as positive; otherwise negative.

## Code
```python
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense

# Small training dataset (short and easy to write)
texts = [
    "i love this movie",
    "this film is amazing",
    "i enjoyed the story",
    "this movie is fantastic",
    "i hate this movie",
    "this film is terrible",
    "i disliked the plot",
    "this movie is boring"
]
labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # 1=positive, 0=negative

# Tokenization and sequence preparation
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
max_len = 6
X = pad_sequences(sequences, maxlen=max_len, padding='post')

# Build RNN model
model = Sequential([
    Embedding(input_dim=1000, output_dim=16, input_length=max_len),
    SimpleRNN(16),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, labels, epochs=30, verbose=0)

# Test sentences
test_texts = ["i love this film", "this movie is terrible"]
test_seq = tokenizer.texts_to_sequences(test_texts)
test_pad = pad_sequences(test_seq, maxlen=max_len, padding='post')

pred = model.predict(test_pad, verbose=0)

for text, p in zip(test_texts, pred):
    sentiment = "Positive" if p[0] > 0.5 else "Negative"
    print(f"Text: {text}")
    print(f"Prediction Score: {p[0]:.4f}")
    print(f"Class: {sentiment}\n")
```

## Expected Output
```text
Text: i love this film
Prediction Score: 0.9xxx
Class: Positive

Text: this movie is terrible
Prediction Score: 0.0xxx
Class: Negative
```

Note: Exact prediction scores may vary slightly in each run due to random weight initialization.
