import json
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
import torch
from sklearn.metrics import accuracy_score

# Function to read the jsonl file and return a DataFrame
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return pd.DataFrame(data)

# Define the dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len  # Define max_len here

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Define the path where the model and tokenizer are saved
model_load_path = './bert_monolingual_model'  # Adjust this path as needed

# Initialize BERT tokenizer and load the trained model
tokenizer = BertTokenizer.from_pretrained(model_load_path)
model = BertForSequenceClassification.from_pretrained(model_load_path)
model.eval()  # Set the model to evaluation mode

# Path to the test data
test_data_path = '../subtaskA/data/subtaskA_dev_monolingual.jsonl'

# Load and preprocess the test data
test_data = load_data(test_data_path)

# Extract texts and labels
test_texts = test_data['text'].tolist()
test_labels = test_data['label'].tolist()

# Create dataset and DataLoader
max_len = 256  # This should be the same value as used in the training script
test_dataset = TextDataset(test_texts, test_labels, tokenizer, max_len)
test_loader = DataLoader(test_dataset, batch_size=16)

# Set device for testing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Evaluation loop
predictions, true_labels = [], []
for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)

    logits = outputs.logits.detach().cpu().numpy()
    label_ids = batch['labels'].to('cpu').numpy()

    predictions.append(logits)
    true_labels.append(label_ids)

# Calculate the accuracy
predictions = np.concatenate(predictions, axis=0)
true_labels = np.concatenate(true_labels, axis=0)
predictions = np.argmax(predictions, axis=1)
accuracy = accuracy_score(true_labels, predictions)

print(f'Accuracy: {accuracy:.2f}')
