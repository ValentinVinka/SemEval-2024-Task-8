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

from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

import os


class DetectDataset(Dataset):
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
    
    
class TextClassifier(nn.Module):
    def __init__(self, base_model):
        global layer1_in, layer1_out, layer2_in, layer2_out    
        super(TextClassifier, self).__init__()

        layer2_in = layer1_out
        layer2_out = 1
        self.bert = base_model
        self.fc1 = nn.Linear(layer1_in, layer1_out)
        self.fc2 = nn.Linear(layer2_in, layer2_out)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        #self.linear = nn.Sequential(
        #    nn.Linear(512, 256),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(256, 128),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(128, 64),
        #    nn.ReLU(inplace=True),
        #    nn.Linear(64, layer2_out)
        #)
        #print('Linear: ', self.linear)
        
        
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
    print("Use device:", device)
    return device

def train(model, train_dataloader, val_dataloader, learning_rate, epochs, model_name):        
    best_val_loss = float('inf')
    early_stopping_threshold_count = 0
    
    
    device = target_device()
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
    device = target_device()
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
    
def create_and_train(train_path, tokenizer, base_model, model_name):
    # Load JSON file with dataset. Perform basic transformations.
    train_df = pd.read_json(train_path, lines=True)
    train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42)

    train_df = train_df.drop(["model", "source"], axis=1)
    val_df = val_df.drop(["model", "source"], axis=1)
    
        
    train_dataloader = DataLoader(DetectDataset(train_df, tokenizer), batch_size=8, shuffle=True, num_workers=0)
    val_dataloader = DataLoader(DetectDataset(val_df, tokenizer), batch_size=8, num_workers=0)

    model = TextClassifier(base_model)


    learning_rate = 1e-5
    epochs = 5
    train(model, train_dataloader, val_dataloader, learning_rate, epochs, model_name)
    
def load_and_evaluate(test_path, prediction_path, tokenizer, model_name):
    test_df = pd.read_json(test_path, lines=True)
    test_df = test_df.drop(["model", "source"], axis=1)
    
    model = torch.load(model_name + ".pt")
    
    predictions_df = pd.DataFrame({'id': test_df['id']})
    test_dataloader = DataLoader(DetectDataset(test_df, tokenizer), batch_size=8, shuffle=False, num_workers=0)
    predictions_df['label'] = get_text_predictions(model, test_dataloader)
    print("predictions=", predictions_df)
    #
    predictions_df.to_json(prediction_path, lines=True, orient='records')
    
    
torch.manual_seed(0)
np.random.seed(0)

absolute_path = os.path.abspath('subtaskA/data')

train_path = absolute_path + '/subtaskA_train_monolingual.jsonl'
test_path = absolute_path + '/subtaskA_dev_monolingual.jsonl'
prediction_path = absolute_path + '/subtaskA_prediction_monolingual_pytorch_distilbert.jsonl'

#layer1_in = 768
#layer1_out = 32 

layer1_in = 1024
layer1_out = 8

BERT_MODEL = "roberta-large"
model_name="roberta-large"
#model_name="best_distilbert"
#BERT_MODEL = "distilbert-base-uncased"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL)
base_model = AutoModel.from_pretrained(BERT_MODEL)
#load_and_evaluate(test_path, prediction_path, tokenizer, model_name)
create_and_train(train_path, tokenizer, base_model, model_name)