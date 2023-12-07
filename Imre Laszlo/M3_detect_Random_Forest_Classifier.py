import jsonlines
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

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

# Reduce the dataset size for faster experimentation (optional)
# train_df, _ = train_test_split(train_df, test_size=0.9, random_state=42)

# Concatenate training and testing data for vectorization
all_data = pd.concat([train_df['text'], test_df['text']])

# Labeling: 0 for human, 1 for AI
train_labels = train_df['label']

# TF-IDF Vectorization with optimized parameters
tfidf_vectorizer = TfidfVectorizer(max_features=5000)  # Adjust max_features as needed
tfidf_vectorizer.fit(all_data)
train_vectors = tfidf_vectorizer.transform(train_df['text'])
test_vectors = tfidf_vectorizer.transform(test_df['text'])

# Initialize a Random Forest Classifier with parallel processing
classifier = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Train the Random Forest Classifier
classifier.fit(train_vectors, train_labels)

# Predict using the Random Forest Classifier
predicted = classifier.predict(test_vectors)

# Evaluate the classifier
print(classification_report(test_df['label'], predicted))
