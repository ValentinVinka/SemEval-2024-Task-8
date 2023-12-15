import jsonlines
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('stopwords')

# Load the JSONL data
def load_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

# Function for text preprocessing
def preprocess_text(text):
    # Lowercasing
    text = text.lower()

    # Tokenization
    tokens = word_tokenize(text)

    # Removing punctuation
    tokens = [token for token in tokens if token not in string.punctuation]

    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Join tokens back into text
    processed_text = ' '.join(tokens)
    return processed_text

# File paths for original data and preprocessed output
train_data_path = r'C:\Master\IA1\Sem1\Semantica_si_Pragmatica_Limbajului_Natural\Proiect\subtaskA_train_monolingual.jsonl'
test_data_path = r'C:\Master\IA1\Sem1\Semantica_si_Pragmatica_Limbajului_Natural\Proiect\subtaskA_dev_monolingual.jsonl'

train_output_path = r'C:\Master\IA1\Sem1\Semantica_si_Pragmatica_Limbajului_Natural\Proiect\subtaskA_train_monolingual_preprocessed.jsonl'
test_output_path = r'C:\Master\IA1\Sem1\Semantica_si_Pragmatica_Limbajului_Natural\Proiect\subtaskA_dev_monolingual_preprocessed.jsonl'

# Load training and testing data
train_data = load_jsonl(train_data_path)
test_data = load_jsonl(test_data_path)

# Apply advanced text preprocessing to training and testing data
for data in [train_data, test_data]:
    for obj in data:
        obj['text'] = preprocess_text(obj['text'])

# Write preprocessed training and testing data to new JSONL files
with jsonlines.open(train_output_path, 'w') as writer:
    for obj in train_data:
        writer.write(obj)

with jsonlines.open(test_output_path, 'w') as writer:
    for obj in test_data:
        writer.write(obj)
