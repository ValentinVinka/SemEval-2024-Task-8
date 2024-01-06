import json
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, AdamW
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
        self.max_len = max_len  # Define max_len here as a fixed value

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


# Initialize BERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
max_len = 256  # Adjust as needed

# Path to the training data
train_data_path = '../subtaskA/data/subtaskA_train_monolingual.jsonl'

# Load and preprocess the training data
train_data = load_data(train_data_path)

# Extract texts and labels
train_texts = train_data['text'].tolist()
train_labels = train_data['label'].tolist()

# Create dataset and DataLoader
train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Set device for training
if torch.cuda.is_available():
    print("Using GPU")
else:
    print("Using CPU")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize BERT model for sequence classification
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model = model.to(device)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

print('Starting training')
# Training loop
model.train()
for epoch in range(3):  # Adjust number of epochs if needed
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        model.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch} completed')

# Save the trained model and tokenizer to a local directory
model_save_path = './bert_monolingual_model'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print("Model training and saving completed.")
