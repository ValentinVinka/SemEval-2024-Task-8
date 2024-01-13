import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset
from torch.optim import Adam, AdamW
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from transformers import AutoTokenizer, AutoModel

from tqdm import tqdm

import nltk
import re
import string
import json
from datetime import datetime
import types
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
class CustomClassifierBase(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomClassifierBase, self).__init__()

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
class CustomClassifierRobertaLarge(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomClassifierRobertaLarge, self).__init__()

        self.bert = pretrained_model
        self.fc1 = nn.Linear(1024, 8)
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
    
#classifier for roberta base,bert, distilbert with 768 neurons on layer 1 and 32 on layer 2 and 8 on layer 3
class CustomClassifierBase3Layers(nn.Module):
    def __init__(self, pretrained_model):
        super(CustomClassifierBase3Layers, self).__init__()

        self.bert = pretrained_model
        self.fc1 = nn.Linear(768, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids=input_ids,
                             attention_mask=attention_mask)[0][:, 0]
        x = self.fc1(bert_out)
        x = self.relu(x)
        
        x = self.fc2(x)
        x = self.relu(x)
        
        x = self.fc3(x)
        x = self.sigmoid(x)

        return x

class DistilbertCustomClassifier(nn.Module):
    def __init__(self,
                 bert_model,
                 num_labels = 1, 
                 bert_hidden_dim=768, 
                 classifier_hidden_dim=32, 
                 dropout=None):
        
        super().__init__()
        self.bert_model = bert_model
        # nn.Identity does nothing if the dropout is set to None
        self.head = nn.Sequential(nn.Linear(bert_hidden_dim, classifier_hidden_dim),
                                  nn.ReLU(),
                                  nn.Dropout(dropout) if dropout is not None else nn.Identity(),
                                  nn.Linear(classifier_hidden_dim, num_labels))
    
    def forward(self, input_ids, attention_mask):
        # feeding the input_ids and masks to the model. These are provided by our tokenizer
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        # obtaining the last layer hidden states of the Transformer
        last_hidden_state = output.last_hidden_state # shape: (batch_size, seq_length, bert_hidden_dim)
        # As I said, the CLS token is in the beginning of the sequence. So, we grab its representation 
        # by indexing the tensor containing the hidden representations
        CLS_token_state = last_hidden_state[:, 0, :]
        # passing this representation through our custom head
        logits = self.head(CLS_token_state)
        return logits
            
def str_to_class(s):
    #if s in globals() and isinstance(globals()[s], types.ClassType):
    return globals()[s]
    

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
    #optimizer = Adam(model.parameters(), lr=learning_rate)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    model = model.to(device)
    criterion = criterion.to(device)

    for epoch in range(epochs):
        total_acc_train = 0
        total_loss_train = 0
        
        model.train()
        
        for train_input, train_label in tqdm(train_dataloader):
            optimizer.zero_grad()
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
                torch.save(model, app_configs['models_path'] + model_name + ".pt")
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
    
  
def get_train_data(train_path, random_seed = 0):
    """
    function to read dataframe with columns
    """

    train_df = pd.read_json(train_path, lines=True)
    

    percentOfData = app_configs['percent_of_data']
    train_df = train_df.sample(int(len(train_df)*percentOfData/100))
    print(len(train_df))
    return train_df    

   
def create_and_train():
    #global app_configs
       
    # Load JSON file with dataset. Perform basic transformations.
    #train_df = pd.read_json(app_configs['train_path'], lines=True)
    train_df = get_train_data(app_configs['train_path'])    
    
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    train_df = train_df.drop(["model", "source"], axis=1)
    val_df = val_df.drop(["model", "source"], axis=1)
    
    target_device()    
    tokenizer, pretrained_model = get_pretrained_model()
        
    train_dataloader = DataLoader(PreprocessDataset(train_df, tokenizer), batch_size=8, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(PreprocessDataset(val_df, tokenizer), batch_size=8, num_workers=0)

    
    classifierClass = str_to_class(app_configs['classifier'])
    myModel = classifierClass(pretrained_model)
    train(myModel, train_dataloader, val_dataloader, app_configs['learning_rate'], app_configs['epochs'], app_configs['model_name'])
    
def load_and_evaluate(model_name = ''):
    #global app_configs
    if (model_name):
        app_configs['model_name'] = model_name
        
    test_df = pd.read_json(app_configs['test_path'], lines=True)
    test_df = test_df.drop(["model", "source"], axis=1)
    
    target_device()
    tokenizer, pretrained_model = get_pretrained_model()
    
    model = torch.load(app_configs['models_path'] + app_configs['model_name'] + ".pt")
    
    predictions_df = pd.DataFrame({'id': test_df['id']})
    test_dataloader = DataLoader(PreprocessDataset(test_df, tokenizer), batch_size=8, shuffle=False, num_workers=0)
    predictions_df['label'] = get_text_predictions(model, test_dataloader)
    #
    predictions_df.to_json(app_configs['prediction_path'], lines=True, orient='records')
    merged_df = predictions_df.merge(test_df, on='id', suffixes=('_pred', '_gold'))
    accuracy = accuracy_score(merged_df['label_gold'], merged_df['label_pred'])
    app_configs['accuracy'] = accuracy
    print("Accuracy:", accuracy)
    print(classification_report(merged_df['label_gold'], merged_df['label_pred']))
    if (model_name == ''): #save app options only when evaluation is called right after training
        save_app_options()

def save_app_options():
    configs = app_configs.copy()
    configs_keys = configs.keys()
    
    keys_2_del = {'tokenizer', 'pretrained_model', 'prediction_path', 'results_path', 'options_path', 'options_path', 'classifier', 'device'}
    for del_key in keys_2_del:
        configs.pop(del_key, None)
           
        
    # Writing to sample.json
    with open(app_configs['options_path'], "w") as outfile:
        json.dump(configs, outfile)
        
# datetime object containing current date and time
start_now = datetime.now()
start_time= start_now.strftime("%Y-%m-%d %H-%M")
timestamp_prefix = start_now.strftime("%Y%m%d%H%M")
print("process start at:", start_time)
torch.manual_seed(0)
np.random.seed(0)

absolute_path = os.path.abspath('subtaskA/data')


distilbert_model_configs1 = {
    'base_model': 'distilbert-base-uncased',
    'classifier': 'CustomClassifierBase',
}

distilbert_model_configs2 = {
    'base_model': 'distilbert-base-uncased',
    'classifier': 'CustomClassifierBase3Layers',
    'learning_rate': 2e-5,
}

robertabase_model_configs1 = {
    'base_model': 'roberta-base',
    'classifier': 'CustomClassifierBase', 
}

robertalarge_model_configs1 = {
    'base_model': 'roberta-large',    
    'classifier': 'CustomClassifierRobertaLarge', 
    'percent_of_data': 100,
}

default_configs = {
    'learning_rate': 1e-5,
    'epochs': 5,
    'task': 'subtaskA_monolingual',
    'timestamp_prefix': timestamp_prefix,
    'train_path': absolute_path + '/subtaskA_train_monolingual.jsonl',
    'test_path': absolute_path + '/subtaskA_dev_monolingual.jsonl',
    'percent_of_data': 100,
    'options_path': absolute_path + '/predictions/'  + 'tests.results.jsonl',
    'models_path':  absolute_path + '/models/',
}


app_configs = default_configs.copy()
app_configs.update(robertabase_model_configs1)

app_configs['model_name'] = app_configs['timestamp_prefix'] + "_" + app_configs['task'] + "_" + app_configs['base_model']
app_configs['prediction_path'] = absolute_path + '/predictions/' + app_configs['model_name'] + '.predictions.jsonl'
app_configs['options_path'] = absolute_path + '/predictions/'  + app_configs['model_name'] + '.options.jsonl'
app_configs['results_path'] = absolute_path + '/predictions/'  + app_configs['model_name'] + '.results.jsonl'

print("Working on pretrained-model:", app_configs['base_model'])

#model names that can be used for evaluation:
#model name roberta-large trained = 202401112145_subtaskA_monolingual_roberta-large
#model name for distilbert-base-uncased trained = 202401120919_subtaskA_monolingual_distilbert-base-uncased - 2 layers

model_for_evaluate=''
create_and_train()
load_and_evaluate(model_for_evaluate)

end_now = datetime.now()
end_time = end_now.strftime("%Y-%m-%d %H-%M")
print("process finished at:", end_time)
running_time = (end_now - start_now).total_seconds()
app_configs['start_time'] = start_time
app_configs['end_time'] = end_time
app_configs['running_time'] = running_time
if (model_for_evaluate == ''): 
    save_app_options()
