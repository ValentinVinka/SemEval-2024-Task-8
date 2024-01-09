import torch
from torch import nn
from torch.utils.data import Dataset
from torch.optim import Adam

from tqdm import tqdm
import numpy as np
import pandas as pd
import re
import nltk
import string

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import FunctionTransformer
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

import pickle
import os

app_configs = {}

class PreprocessDataset(Dataset):
    def __init__(self, dataframe, tokenizer):
        texts = dataframe.text.values.tolist()

        texts = [self._preprocess(text) for text in texts]

        #self._print_random_samples(texts)

        self.texts = [tokenizer(text, padding='max_length',
                                max_length=150,
                                truncation=True,
                                return_tensors="pt")
                      for text in texts]

        if 'label' in dataframe:
            self.labels = dataframe.label.values.tolist()

    def _print_random_samples(self, texts):
        np.random.seed(42)
        random_entries = np.random.randint(0, len(texts), 5)

        for i in random_entries:
            print(f"Entry {i}: {texts[i]}")

        print()

    def _preprocess(self, text):
        text = self._remove_amp(text)
        text = self._remove_links(text)
        text = self._remove_hashes(text)
        #text = self._remove_retweets(text)
        text = self._remove_mentions(text)
        text = self._remove_multiple_spaces(text)

        #text = self._lowercase(text)
        text = self._remove_punctuation(text)
        #text = self._remove_numbers(text)

        text_tokens = self._tokenize(text)
        text_tokens = self._stopword_filtering(text_tokens)
        #text_tokens = self._stemming(text_tokens)
        text = self._stitch_text_tokens_together(text_tokens)

        return text.strip()


    def _remove_amp(self, text):
        return text.replace("&amp;", " ")

    def _remove_mentions(self, text):
        return re.sub(r'(@.*?)[\s]', ' ', text)
    
    def _remove_multiple_spaces(self, text):
        return re.sub(r'\s+', ' ', text)

    def _remove_retweets(self, text):
        return re.sub(r'^RT[\s]+', ' ', text)

    def _remove_links(self, text):
        return re.sub(r'https?:\/\/[^\s\n\r]+', ' ', text)

    def _remove_hashes(self, text):
        return re.sub(r'#', ' ', text)

    def _stitch_text_tokens_together(self, text_tokens):
        return " ".join(text_tokens)

    def _tokenize(self, text):
        return nltk.word_tokenize(text, language="english")

    def _stopword_filtering(self, text_tokens):
        stop_words = nltk.corpus.stopwords.words('english')

        return [token for token in text_tokens if token not in stop_words]

    def _stemming(self, text_tokens):
        porter = nltk.stem.porter.PorterStemmer()
        return [porter.stem(token) for token in text_tokens]

    def _remove_numbers(self, text):
        return re.sub(r'\d+', ' ', text)

    def _lowercase(self, text):
        return text.lower()

    def _remove_punctuation(self, text):
        return ''.join(character for character in text if character not in string.punctuation)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]

        label = -1
        if hasattr(self, 'labels'):
            label = self.labels[idx]

        return text, label
    
    
#classifier for roberta base,bert, distilbert with 768 neurons on layer 1 and 32 on layer 2
class CustomClassifier1(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomClassifier1, self).__init__()

        self.bert = pretrained_model
        self.fc1 = nn.Linear(768, 32)
        self.fc2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(bert_out)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x

#classifier for roberta large with 1024 neurons on layer 1 and 8 on layer 2
class CustomClassifier2(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomClassifier1, self).__init__()

        self.bert = pretrained_model
        self.fc1 = nn.Linear(1024, 32)
        self.fc2 = nn.Linear(8, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(bert_out)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.sigmoid(x)

        return x
    
def target_device():
    #gpu support for Mac
    use_mps = torch.backends.mps.is_available()
    use_cuda = torch.cuda.is_available()        
    device = torch.device("cuda" if use_cuda else "mps" if use_mps else "cpu")    
    app_configs['device'] = device
    print("Use device:", device)
    return device

def train(model, train_dataloader, val_dataloader, learning_rate, epochs, model_name):        
    best_val_loss = float('inf')
    early_stopping_threshold_count = 0
    
    
    device = app_configs['device']
    criterion = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        
        model.train()
        
        for train_input, train_label in tqdm(train_dataloader):
            attention_mask = train_input['attention_mask'].to(device)
            input_ids = train_input['input_ids'].squeeze(1).to(device)

            train_label = train_label.to(device)

            output = model(input_ids, attention_mask)

            loss = criterion(output, train_label.float().unsqueeze(1))

            total_loss_train += loss.item()

            acc = ((output >= 0.5).int() == train_label.unsqueeze(1)).sum().item()
            total_acc_train += acc

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        with torch.no_grad():
            total_acc_val = 0
            total_loss_val = 0
            
            model.eval()
            
            for val_input, val_label in tqdm(val_dataloader):
                attention_mask = val_input['attention_mask'].to(device)
                input_ids = val_input['input_ids'].squeeze(1).to(device)

                val_label = val_label.to(device)

                output = model(input_ids, attention_mask)

                loss = criterion(output, val_label.float().unsqueeze(1))

                total_loss_val += loss.item()

                acc = ((output >= 0.5).int() == val_label.unsqueeze(1)).sum().item()
                total_acc_val += acc
            
            print(f'Epochs: {epoch + 1} '
                  f'| Train Loss: {total_loss_train / len(train_dataloader): .3f} '
                  f'| Train Accuracy: {total_acc_train / (len(train_dataloader.dataset)): .3f} '
                  f'| Val Loss: {total_loss_val / len(val_dataloader): .3f} '
                  f'| Val Accuracy: {total_acc_val / len(val_dataloader.dataset): .3f}')
            
            if best_val_loss > total_loss_val:
                best_val_loss = total_loss_val
                torch.save(model, model_name + ".pt")
                print("Saved model")
                early_stopping_threshold_count = 0
            else:
                early_stopping_threshold_count += 1
                
            if early_stopping_threshold_count >= 1:
                print("Early stopping")
                break

def get_text_predictions(model, loader):
    device = app_configs['device']
    model = model.to(device)
    
    results_predictions = []
    with torch.no_grad():
        model.eval()
        for data_input, _ in tqdm(loader):
            attention_mask = data_input['attention_mask'].to(device)
            input_ids = data_input['input_ids'].squeeze(1).to(device)


            output = model(input_ids, attention_mask)
            
            output = (output > 0.5).int()
            results_predictions.append(output)
    
    return torch.cat(results_predictions).cpu().detach().numpy()
    
def get_pretrained_model():
    global app_configs
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(app_configs['base_model'])
    pretrained_model = AutoModel.from_pretrained(app_configs['base_model'])
    
    app_configs['tokenizer'] = tokenizer
    app_configs['pretrained_model'] = pretrained_model
    return tokenizer, pretrained_model
    
#Funcție pentru a obține reprezentări DistilBERT pentru un text
def get_distilbert_embedding(text):
    tokens = app_configs['tokenizer'](text, return_tensors='pt')
    # Mutarea datelor pe GPU
    tokens = {key: value.to(app_configs['device']) for key, value in tokens.items()}
    with torch.no_grad():
        outputs = app_configs['pretrained_model'](**tokens)
    return outputs.last_hidden_state.mean(dim=1).squeeze()
   
def get_data(train_path, test_path, random_seed):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    test_df = pd.read_json(test_path, lines=True)
    
    train_df, val_df = train_test_split(train_df, test_size=0.2, stratify=train_df['label'], random_state=random_seed)

    percentOfData = 10
    train_df = train_df.sample(int(len(train_df)*percentOfData/100))
    val_df = val_df.sample(int(len(val_df)*percentOfData/100))
    test_df = test_df.sample(int(len(test_df)*percentOfData/100))
    print(len(train_df))
    print(len(val_df))
    print(len(test_df))
    return train_df, val_df, test_df    

def pipeline_train(traid_df):
    # Exemplu de date
    texts = traid_df['text']
    labels = traid_df['label']

    # Divizați setul de date în antrenare și testare
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    # Construiți un pipeline cu TfidfVectorizer și un clasificator
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('tfidf', TfidfVectorizer()),
            ('distilbert', FunctionTransformer(get_distilbert_embedding, validate=False))
        ])),
        ('classifier', LogisticRegression())
    ])
    # Mutarea întregului pipeline pe GPU
    #pipeline.named_steps['classifier'].to(app_configs['device'])

    # Antrenați pipeline-ul pe datele de antrenare
    pipeline.fit(X_train, y_train)   
    # Salvați modelul într-un fișier
    with open(app_configs['model_name'] + ".pipeline", 'wb') as file:
        pickle.dump(pipeline, file)
        
    #joblib.dump(pipeline, app_configs['model_name'] + ".pipeline")
    
def create_and_train():
    #global app_configs
       
    # Load JSON file with dataset. Perform basic transformations.
    train_df = pd.read_json(app_configs['train_path'], lines=True)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    train_df = train_df.drop(["model", "source"], axis=1)
    val_df = val_df.drop(["model", "source"], axis=1)
    
    target_device()    
    tokenizer, pretrained_model = get_pretrained_model()
        
    train_dataloader = DataLoader(PreprocessDataset(train_df, tokenizer), batch_size=8, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(PreprocessDataset(val_df, tokenizer), batch_size=8, num_workers=0)

    if (app_configs['pipeline']):
        pipeline_train(train_df)
    else:
        myModel = app_configs['classifier'](pretrained_model)
        train(myModel, train_dataloader, val_dataloader, app_configs['learning_rate'], app_configs['epochs'], app_configs['model_name'])
    
def load_and_evaluate():
    #global app_configs
    
    test_df = pd.read_json(app_configs['test_path'], lines=True)
    test_df = test_df.drop(["model", "source"], axis=1)
    
    target_device()
    tokenizer, pretrained_model = get_pretrained_model()
    
    model = torch.load(app_configs['model_name'] + ".pt")
    
    predictions_df = pd.DataFrame({'id': test_df['id']})
    test_dataloader = DataLoader(PreprocessDataset(test_df, tokenizer), batch_size=8, shuffle=False, num_workers=0)
    predictions_df['label'] = get_text_predictions(model, test_dataloader)
    #
    predictions_df.to_json(app_configs['prediction_path'], lines=True, orient='records')
    merged_df = predictions_df.merge(test_df, on='id', suffixes=('_pred', '_gold'))
    accuracy = accuracy_score(merged_df['label_gold'], merged_df['label_pred'])
    print("Accuracy:", accuracy)
    
torch.manual_seed(0)
np.random.seed(0)

absolute_path = os.path.abspath('subtaskA/data')


distilbert_model_configs1 = {
    'base_model': 'distilbert-base-uncased',
    'classifier': CustomClassifier1,
    'pipeline': 1,
    'model_prefix': '3mdetect_pipeline'
}

robertabase_model_configs1 = {
    'base_model': 'roberta-large',
    'classifier': CustomClassifier1, 
}

robertalarge_model_configs1 = {
    'base_model': 'roberta-large',
    
    'classifier': CustomClassifier2, 
}

default_configs = {
    'model_prefix': '3mdetect',
    'learning_rate': 1e-5,
    'epochs': 5
}

app_configs = default_configs.copy()
app_configs.update(distilbert_model_configs1)

app_configs['model_name'] = app_configs['model_prefix'] + "_" + app_configs['base_model']
app_configs['train_path'] = absolute_path + '/subtaskA_train_monolingual.jsonl'
app_configs['test_path'] = absolute_path + '/subtaskA_dev_monolingual.jsonl'
app_configs['prediction_path'] = absolute_path + '/predictions/subtaskA_prediction_monolingual_' + app_configs['model_name'] + '.jsonl'

print("Working on pretrained-model:", app_configs['base_model'])
create_and_train()
#load_and_evaluate()