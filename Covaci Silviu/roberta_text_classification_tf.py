import numpy as np
import regex as re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics
import math
import os

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tokenizers
from transformers import RobertaTokenizer, TFRobertaModel

from collections import Counter

import warnings
warnings.filterwarnings("ignore")

import torch

torch.device("mps")

print("is ava=", torch.cuda.is_available())
exit()


def processor_name():
    import cpuinfo
    cpudata = cpuinfo.get_cpu_info()['brand_raw']
    cpuname = cpudata.split(" ")[1]
    return cpuname

def category_to_number(categories, y_data):
    # Transform categories into numbers
    category_to_id = {}
    category_to_name = {}
    
    for index, c in enumerate(y_data):
        if c in category_to_id:
            category_id = category_to_id[c]
        else:
            category_id = len(category_to_id)
            category_to_id[c] = category_id
            category_to_name[category_id] = c
    
        y_data[index] = category_id
        
        
    return y_data    

def roberta_encode(texts, tokenizer):
    ct = len(texts)
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32') # Not used in text classification

    for k, text in enumerate(texts):
        # Tokenize
        tok_text = tokenizer.tokenize(text)
        
        # Truncate and convert tokens to numerical IDs
        enc_text = tokenizer.convert_tokens_to_ids(tok_text[:(MAX_LEN-2)])
        
        input_length = len(enc_text) + 2
        input_length = input_length if input_length < MAX_LEN else MAX_LEN
        
        # Add tokens [CLS] and [SEP] at the beginning and the end
        input_ids[k,:input_length] = np.asarray([0] + enc_text + [2], dtype='int32')
        
        # Set to 1s in the attention input
        attention_mask[k,:input_length] = 1

    return {
        'input_word_ids': input_ids,
        'input_mask': attention_mask,
        'input_type_ids': token_type_ids
    }
    
def build_model(n_categories):
    processor = processor_name()
    print("processor name:" + processor)
    
    with strategy.scope():
        input_word_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_word_ids')
        input_mask = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_mask')
        input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_type_ids')

        # Import RoBERTa model from HuggingFace
        roberta_model = TFRobertaModel.from_pretrained(MODEL_NAME)
        x = roberta_model(input_word_ids, attention_mask=input_mask, token_type_ids=input_type_ids)

        # Huggingface transformers have multiple outputs, embeddings are the first one,
        # so let's slice out the first position
        x = x[0]

        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.Dense(n_categories, activation='softmax')(x)

        optimiser = tf.keras.optimizers.Adam(learning_rate=1e-5)
        if (processor == 'M1' or processor =='M2'):
            optimiser = tf.keras.optimizers.legacy.Adam(learning_rate=1e-5)
            
        #lossed = 'sparse_categorical_crossentropy'
        losses={'encode_decode': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'question_answering': tf.keras.losses.mean_squared_error,
            'reasoning': tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            'consistency': tf.keras.losses.mean_squared_error}
            
        model = tf.keras.Model(inputs=[input_word_ids, input_mask, input_type_ids], outputs=x)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-5),            
            loss=losses,
            metrics=['accuracy'])

        return model
        
def train_model(model, X_train, y_train, X_test, y_test):
    with strategy.scope():
        print('Training...')
        history = model.fit(X_train,
                            y_train,
                            epochs=EPOCHS,
                            batch_size=BATCH_SIZE,
                            verbose=1,
                            validation_data=(X_test, y_test))        
    
    # This plot will look much better if we train models with more epochs, but anyway here is
    plt.figure(figsize=(10, 10))
    plt.title('Accuracy')

    xaxis = np.arange(len(history.history['accuracy']))
    plt.plot(xaxis, history.history['accuracy'], label='Train set')
    plt.plot(xaxis, history.history['val_accuracy'], label='Validation set')
    plt.legend()    
    return model

def plot_confusion_matrix(X_test, y_test, model):
    y_pred = model.predict(X_test)
    y_pred = [np.argmax(i) for i in model.predict(X_test)]

    con_mat = tf.math.confusion_matrix(labels=y_test, predictions=y_pred).numpy()

    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    label_names = list(range(len(con_mat_norm)))

    con_mat_df = pd.DataFrame(con_mat_norm,
                              index=label_names, 
                              columns=label_names)

    figure = plt.figure(figsize=(10, 10))
    sns.heatmap(con_mat_df, cmap=plt.cm.Blues, annot=True)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
# Detect hardware, return appropriate distribution strategy (you can see that it is pretty easy to set up).
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is set (always set in Kaggle)
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print('Running on TPU ', tpu.master())
except ValueError:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print('Number of replicas:', strategy.num_replicas_in_sync)

MODEL_NAME = 'roberta-base'
MAX_LEN = 256
ARTIFACTS_PATH = '../artifacts/'
ARTIFACTS_PATH = os.path.abspath('subtaskA/data')
print("ARTIFACTS_PATH=", ARTIFACTS_PATH)


BATCH_SIZE = 8 * strategy.num_replicas_in_sync
EPOCHS = 3

absolute_path = os.path.abspath('subtaskA/data')

train_path = absolute_path + '/subtaskA_train_monolingual.jsonl'
test_path = absolute_path + '/subtaskA_dev_monolingual.jsonl'
prediction_path = absolute_path + '/subtaskA_prediction_monolingual.jsonl'

    
if not os.path.exists(ARTIFACTS_PATH):
    os.makedirs(ARTIFACTS_PATH)
    
# Load JSON file with dataset. Perform basic transformations.
df = pd.read_json(train_path, lines=True)
X_data = df[['text']].to_numpy().reshape(-1)
y_data = df[['model']].to_numpy().reshape(-1)    
#categories = df[['label']].values.reshape(-1)
print("y_data=" , y_data)
    
n_texts = len(X_data)
print('Texts in dataset: %d' % n_texts)

categories = df['model'].unique()
n_categories = len(categories)
print('Number of categories: %d' % n_categories)
 
y_data = category_to_number(categories, y_data);

# Split into train and test datasets
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3, random_state=777) # random_state to reproduce results

# Import tokenizer from HuggingFace
tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

X_train = roberta_encode(X_train, tokenizer)
X_test = roberta_encode(X_test, tokenizer)

y_train = np.asarray(y_train, dtype='int32')
y_test = np.asarray(y_test, dtype='int32')

with strategy.scope():
    model = build_model(n_categories)
    model.summary()
    
model = train_model(model, X_train, y_train, X_test, y_test)

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1] * 100))


plot_confusion_matrix(X_test, y_test, model)

exit()
