import jsonlines
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense

# Load the JSONL data
def load_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

# Load training and testing data
train_data = load_jsonl(r'C:\Master\IA1\Sem1\Semantica_si_Pragmatica_Limbajului_Natural\Proiect\subtaskA_train_monolingual.jsonl')
test_data = load_jsonl(r'C:\Master\IA1\Sem1\Semantica_si_Pragmatica_Limbajului_Natural\Proiect\subtaskA_dev_monolingual.jsonl')

# Convert JSONL data to DataFrame
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

# Concatenate training and testing data for text processing
all_data = pd.concat([train_df['text'], test_df['text']])

# Labeling: 0 for human, 1 for AI
encoder = LabelEncoder()
train_labels = encoder.fit_transform(train_df['label'])
test_labels = encoder.transform(test_df['label'])

# Tokenize text data and convert to sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(all_data)
train_sequences = tokenizer.texts_to_sequences(train_df['text'])
test_sequences = tokenizer.texts_to_sequences(test_df['text'])

# Pad sequences to a fixed length
max_sequence_length = 100  # Set the maximum sequence length
train_data_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_sequence_length)
test_data_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_sequence_length)

# Build and compile the neural network model
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size
embedding_dim = 50  # Embedding dimension

model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_sequence_length),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(len(encoder.classes_), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the neural network model
model.fit(train_data_padded, train_labels, epochs=5, batch_size=64, validation_split=0.1)

# Evaluate the model on test data
predicted_probabilities = model.predict(test_data_padded)
predicted_labels = np.argmax(predicted_probabilities, axis=1)

# Decode labels to original classes
predicted_labels = encoder.inverse_transform(predicted_labels)

# Evaluate the classifier
print(classification_report(test_df['label'], predicted_labels))
