import jsonlines
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score
#from sklearn.externals import joblib
import os
import pickle

# Load the JSONL data
def load_jsonl(file_path):
    data = []
    with jsonlines.open(file_path) as reader:
        for obj in reader:
            data.append(obj)
    return data

def get_vectorised_data(train_df, test_df):
     # Adjusting TfidfVectorizer parameters
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2))  # Using unigrams and bigrams
    # You can also explore other parameters like max_features, min_df, max_df, etc.

    # Concatenate training and testing data for vectorization
    all_data = pd.concat([train_df['text'], test_df['text']])
    # Fit and transform using the updated vectorizer
    tfidf_vectorizer.fit(all_data)
    train_vectors = tfidf_vectorizer.transform(train_df['text'])
    test_vectors = tfidf_vectorizer.transform(test_df['text'])
    return train_vectors, test_vectors
    
   
def train_model(train_df, test_df, model_path):
   

    # Labeling: 0 for human, 1 for AI
    train_labels = train_df['label']

    train_vectors, test_vectors = get_vectorised_data(train_df, test_df)
    
    # Train a simple Linear SVM classifier
    classifier = LinearSVC(dual=True)
    classifier.fit(train_vectors, train_labels)
    
    # Salvați modelul într-un fișier
    #joblib.dump(classifier, 'm3detect_mgram.joblib')
    # Salvați modelul într-un fișier
    with open(model_path, 'wb') as file:
        pickle.dump(classifier, file)
    print("Model trained!")    
        
def test_model(train_df, test_df, model_path, output_path):            
    train_vectors, test_vectors = get_vectorised_data(train_df, test_df)
                
    # Pentru a încărca modelul din fișier
    with open(model_path, 'rb') as file:
        classifier = pickle.load(file)
    
    predictions_df = pd.DataFrame({'id': test_df['id']})
    predictions_df['label'] = classifier.predict(test_vectors)
    predictions_df.to_json(prediction_path, lines=True, orient='records')
    
    merged_df = predictions_df.merge(test_df, on='id', suffixes=('_pred', '_gold'))
    accuracy = accuracy_score(merged_df['label_gold'], merged_df['label_pred'])
    print("Accuracy:", accuracy)
            
    # Evaluate the classifier
    print(classification_report(test_df['label'], predictions_df['label']))
    
    return predictions_df

absolute_path = os.path.abspath('subtaskA/data')
    
train_path = absolute_path + '/subtaskA_train_monolingual.jsonl'
test_path = absolute_path + '/subtaskA_dev_monolingual.jsonl'
prediction_path = absolute_path + '/subtaskA_prediction_monolingual_ngram.jsonl'

# Load training and testing data
train_data = load_jsonl(train_path)
test_data = load_jsonl(test_path)

# Convert JSONL data to DataFrame
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)

model_path = 'm3detect_mgram.model.pkl'
prediction_path = absolute_path + '/subtaskA_prediction_monolingual3.jsonl'

train_model(train_df, test_df, model_path)
test_model(train_df, test_df, model_path, prediction_path)

print("done!")

print(model_path)
exit()


