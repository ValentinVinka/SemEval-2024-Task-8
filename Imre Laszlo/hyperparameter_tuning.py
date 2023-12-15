import jsonlines
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
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

# Concatenate training and testing data for vectorization
all_data = pd.concat([train_df['text'], test_df['text']])

# Labeling: 0 for human, 1 for AI
train_labels = train_df['label']

# TF-IDF Vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(all_data)
train_vectors = tfidf_vectorizer.transform(train_df['text'])
test_vectors = tfidf_vectorizer.transform(test_df['text'])

# Define the parameters grid
param_grid = {'C': [0.1, 1, 10, 100]}  # Define the values for the regularization parameter

# Perform Grid Search
grid_search = GridSearchCV(LinearSVC(), param_grid, cv=5)
grid_search.fit(train_vectors, train_labels)

# Get the best estimator
best_classifier = grid_search.best_estimator_

# Train the classifier using the best parameters
best_classifier.fit(train_vectors, train_labels)

# Predict using the optimized classifier
predicted = best_classifier.predict(test_vectors)

# Evaluate the optimized classifier
print(classification_report(test_df['label'], predicted))
