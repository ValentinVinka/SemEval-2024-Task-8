from datasets import Dataset
import pandas as pd
import evaluate
import numpy as np
import torch as torch
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator

from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer, AutoConfig, set_seed
from transformers import RobertaTokenizer, RobertaModel
import os
from sklearn.model_selection import train_test_split
from scipy.special import softmax
import argparse
import logging
import re
import string

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')

punctuationExclude = string.punctuation
tagsRE = re.compile(r'<[^>]+>')
urlsRE = re.compile(r'http\S+|www\.S+')
stopWords = stopwords.words('english')

# Model with classifier layers on top of RoBERTa
class ROBERTAClassifier(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(ROBERTAClassifier, self).__init__()
        
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.d1 = torch.nn.Dropout(dropout_rate)
        self.l1 = torch.nn.Linear(768, 64)
        self.bn1 = torch.nn.LayerNorm(64)
        self.d2 = torch.nn.Dropout(dropout_rate)
        self.l2 = torch.nn.Linear(64, 2)
        
    def forward(self, input_ids, attention_mask):
        _, x = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        x = self.d1(x)
        x = self.l1(x)
        x = self.bn1(x)
        x = torch.nn.Tanh()(x)
        x = self.d2(x)
        x = self.l2(x)
        
        return x 
    
def remove_tags(text):
    return tagsRE.sub('', text)

def remove_url(text):
    return urlsRE.sub('', text)

def remove_punctuation(text):
    for char in punctuationExclude:
        text = text.replace(char, '')
    return text

def remove_stop_words(text):
    tokens = nltk.word_tokenize(text)
    words_without_stop_word = []
    for word in tokens:
        if (word in stopWords):
            continue
        else:
            words_without_stop_word.append(word)
            
    return ' '.join(words_without_stop_word)

def preprocess_text(str):
    str = str.lower()
    str = remove_tags(str)
    str = remove_url(str)
    #str = remove_punctuation(str)
    #str = remove_stop_words(str)
    return str
    
def preprocess_function(examples, **fn_kwargs):
    #examples["text"].str.lower() 
    #print(examples['text'][0]);
    
    examples['text'] = [preprocess_text(x) for x in examples['text']]
    
    #print("-------------------------------------------")
    #print(examples['text'][0]);
    #examples["text"].apply(lambda x:x.lower())
    #exit()
    #encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    return fn_kwargs['tokenizer'](examples["text"], truncation=True, padding=True)


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

def compute_metrics(eval_pred):

    f1_metric = evaluate.load("f1")

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    results = {}
    results.update(f1_metric.compute(predictions=predictions, references = labels, average="micro"))

    return results

def fine_tune_roberta(train_df, valid_df, checkpoints_path, id2label, label2id, model):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    # Set tokenizer hyperparameters.
    MAX_SEQ_LEN = 256
    BATCH_SIZE = 16
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)


    # Define columns to read.
    label_field = Field(sequential=False, use_vocab=False, batch_first=True)
    text_field = Field(use_vocab=False, 
                   tokenize=tokenizer.encode, 
                   include_lengths=False, 
                   batch_first=True,
                   fix_length=MAX_SEQ_LEN, 
                   pad_token=PAD_INDEX, 
                   unk_token=UNK_INDEX)

fields = {'titletext' : ('titletext', text_field), 'label' : ('label', label_field)}


# Read preprocessed CSV into TabularDataset and split it into train, test and valid.
train_data, valid_data, test_data = TabularDataset(path=f"{data_path}/prep_news.csv", 
                                                   format='CSV', 
                                                   fields=fields, 
                                                   skip_header=False).split(split_ratio=[0.70, 0.2, 0.1], 
                                                                            stratified=True, 
                                                                            strata_field='label')

# Create train and validation iterators.
train_iter, valid_iter = BucketIterator.splits((train_data, valid_data),
                                               batch_size=BATCH_SIZE,
                                               device=device,
                                               shuffle=True,
                                               sort_key=lambda x: len(x.titletext), 
                                               sort=True, 
                                               sort_within_batch=False)

# Test iterator, no shuffling or sorting required.
test_iter = Iterator(test_data, batch_size=BATCH_SIZE, device=device, train=False, shuffle=False, sort=False)
        
        
def fine_tune(train_df, valid_df, checkpoints_path, id2label, label2id, model):
        
    
    # pandas dataframe to huggingface Dataset
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)
    
    
    # get tokenizer and model from huggingface
    tokenizer = AutoTokenizer.from_pretrained(model)     # put your model here
    model = AutoModelForSequenceClassification.from_pretrained(
       model, num_labels=len(label2id), id2label=id2label, label2id=label2id    # put your model here
    )
    
    # tokenize data for train/valid
    tokenized_train_dataset = train_dataset.map(preprocess_function, batched=True, fn_kwargs={'tokenizer': tokenizer})
    tokenized_valid_dataset = valid_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})


    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


    # create Trainer 
    training_args = TrainingArguments(
        output_dir=checkpoints_path,
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_valid_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("START TRAIN")
    trainer.train()

    # save best model
    best_model_path = checkpoints_path+'/best/'
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    

    trainer.save_model(best_model_path)
    print("TRAINED MODEL SAVED!")



def test(test_df, model_path, id2label, label2id):
            
    # load tokenizer from saved model 
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # load best model
    model = AutoModelForSequenceClassification.from_pretrained(
       model_path, num_labels=len(label2id), id2label=id2label, label2id=label2id
    )
            
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_test_dataset = test_dataset.map(preprocess_function, batched=True,  fn_kwargs={'tokenizer': tokenizer})
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # create Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    # get logits from predictions and evaluate results using classification report
    predictions = trainer.predict(tokenized_test_dataset)
    prob_pred = softmax(predictions.predictions, axis=-1)
    preds = np.argmax(predictions.predictions, axis=-1)
    metric = evaluate.load("bstrai/classification_report")
    results = metric.compute(predictions=preds, references=predictions.label_ids)
    
    # return dictionary of classification report
    return results, preds

def downloadHuggingFaceRepo(model):
    from huggingface_hub import snapshot_download
    
    absolute_path = os.path.abspath('pretrained_models/' + model)
    #print(absolute_path)
    #exit()    
    snapshot_download(repo_id=model, local_dir=absolute_path)

def downloadHuggingFaceModel(model):    
    tokenizer = AutoTokenizer.from_pretrained(model)     # put your model here    
    config = AutoConfig.from_pretrained(model)
    absolute_path = os.path.abspath('pretrained_models/' + model)
    tokenizer.save_pretrained(absolute_path)
    config.save_pretrained(absolute_path)
    vocab_path = absolute_path + "/vocab/"
    tokenizer.save_vocabulary(vocab_path)
    
    print("tokenizer saved!")
    
    
def readArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_file_path", "-tr", required=True, help="Path to the train file.", type=str)
    parser.add_argument("--test_file_path", "-t", required=True, help="Path to the test file.", type=str)
    parser.add_argument("--subtask", "-sb", required=True, help="Subtask (A or B).", type=str, choices=['A', 'B'])
    parser.add_argument("--model", "-m", required=True, help="Transformer to train and test", type=str)
    parser.add_argument("--prediction_file_path", "-p", required=True, help="Path where to save the prediction file.", type=str)

    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    #downloadHuggingFaceModel('xlm-roberta-base')
    #exit()   
    #args = readArgs()    
    
    #app parameters
    absolute_path = os.path.abspath('subtaskA/data')
    train_path = absolute_path + '/subtaskA_train_monolingual.jsonl'
    test_path = absolute_path + '/subtaskA_dev_monolingual.jsonl'
    prediction_path = absolute_path + '/subtaskA_prediction_monolingual2.jsonl'
    subtask='A'
    #model='xlm-roberta-large'
    model='xlm-roberta-base'
    random_seed = 1
    
    if not os.path.exists(train_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    
    if not os.path.exists(test_path):
        logging.error("File doesnt exists: {}".format(train_path))
        raise ValueError("File doesnt exists: {}".format(train_path))
    

    if subtask == 'A':
        id2label = {0: "human", 1: "machine"}
        label2id = {"human": 0, "machine": 1}
    elif subtask == 'B':
        id2label = {0: 'human', 1: 'chatGPT', 2: 'cohere', 3: 'davinci', 4: 'bloomz', 5: 'dolly'}
        label2id = {'human': 0, 'chatGPT': 1,'cohere': 2, 'davinci': 3, 'bloomz': 4, 'dolly': 5}
    else:
        logging.error("Wrong subtask: {}. It should be A or B".format(train_path))
        raise ValueError("Wrong subtask: {}. It should be A or B".format(train_path))

    set_seed(random_seed)

    #get data for train/dev/test sets
    train_df, valid_df, test_df = get_data(train_path, test_path, random_seed)
    
    # train detector model
    #fine_tune(train_df, valid_df, f"{model}/subtask{subtask}/{random_seed}", id2label, label2id, model)
    #exit()
    
    # test detector model
    results, predictions = test(test_df, f"{model}/subtask{subtask}/{random_seed}/best/", id2label, label2id)
    
    logging.info(results)
    predictions_df = pd.DataFrame({'id': test_df['id'], 'label': predictions})
    predictions_df.to_json(prediction_path, lines=True, orient='records')
