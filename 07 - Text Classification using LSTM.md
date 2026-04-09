# Program 7: Text Classification using LSTM

## Aim
To implement text classification using an LSTM network and classify text as positive or negative sentiment.

## Theory
LSTM (Long Short-Term Memory) is a type of recurrent neural network designed to learn long-term dependencies in sequence data. In text classification, words are converted to tokens, padded to equal length, passed through an embedding layer, and then processed by an LSTM layer. The final dense layer with sigmoid activation gives binary class probability.

Workflow:
1. Prepare labeled text data.
2. Tokenize and pad sequences.
3. Build LSTM model.
4. Train model on labeled data.
5. Predict class for new text.

For binary output:

$$\hat{y}=\sigma(z)$$

If $\hat{y} > 0.5$, the text is classified as positive; otherwise negative.

## Code
```python
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# Small text dataset (1 = positive, 0 = negative)
texts = [
    "this movie is excellent",
    "i love this film",
    "the story is very good",
    "what a fantastic performance",
    "this movie is bad",
    "i hate this film",
    "the story is very boring",
    "what a terrible performance"
]
labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])

# Tokenization
vocab_size = 1000
max_len = 6
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)

sequences = tokenizer.texts_to_sequences(texts)
X = pad_sequences(sequences, maxlen=max_len, padding='post')

# Build LSTM model
model = Sequential()
model.add(Embedding(vocab_size, 16, input_length=max_len))
model.add(LSTM(16))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X, labels, epochs=30, verbose=0)

# Test samples
test_texts = ["i love this movie", "this film is terrible"]
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
Text: i love this movie
Prediction Score: 0.9xxx
Class: Positive

Text: this film is terrible
Prediction Score: 0.0xxx
Class: Negative
```

Note: Prediction scores can vary slightly each run due to random initialization and training behavior.
