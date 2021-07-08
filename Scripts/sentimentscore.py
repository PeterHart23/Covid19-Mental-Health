import os
import sys
from pathlib import Path

import pickle
from pathlib import Path
import json
from IPython.display import display



from tensorflow.keras.layers import Input, Embedding, GRU
from tensorflow.keras.layers import Dropout, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import preprocess
from tqdm import tqdm
import pandas as pd
import numpy as np
import re

tokenizer_file = Path('Datasets/sentiment140/tokenizer.pickle').resolve()
with tokenizer_file.open('rb') as file:
    tokenizer = pickle.load(file)

input_dim = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
embedding_dim = 200
input_length = 100
gru_units = 128
gru_dropout = 0.1
recurrent_dropout = 0.1
dropout = 0.1


model = Sequential()
model.add(Embedding(
    input_dim=input_dim,
    output_dim=embedding_dim,
    input_shape=(input_length,)
))

model.add(Bidirectional(GRU(
    gru_units,
    return_sequences=True,
    dropout=gru_dropout,
    recurrent_dropout=recurrent_dropout
)))
model.add(GlobalMaxPooling1D())
model.add(Dense(32, activation='relu'))
model.add(Dropout(dropout))

model.add(Dense(1, activation='sigmoid'))

weights_path = Path('../models/gru_model.h5').resolve()
model.load_weights(weights_path.as_posix())

relations_path = Path('query_relations.json')
with relations_path.open('r') as file:
    relations = json.load(file)

dataset_dir = Path('Datasets/tweepy').resolve()

data_dict = {}

query_dict = {
    'query': [],
    'mean': [],
    'max': [],
    'min': [],
    'std': [],
    'count': [],
    'emotion': []
}

dir_files = os.listdir(dataset_dir)
del dir_files[-1]
print(dir_files)

with tqdm(total=len(dir_files)) as t:
    for filename in dir_files:
        dataset = pd.read_csv(os.path.join(dataset_dir, filename))
        cleaned_texts = preprocess(dataset.text, quiet=True)

        query = re.findall(r'(#[^.]+|:.+:)', filename)[0]

        predict_sequences = [text.split() for text in cleaned_texts]
        list_tokenized_predict = tokenizer.texts_to_sequences(predict_sequences)
        x_predict = pad_sequences(list_tokenized_predict, maxlen=100)

        result = model.predict(x_predict)

        emotion = relations[query]
        query_dict['query'].append(query)
        query_dict['mean'].append(np.mean(result))
        query_dict['max'].append(np.amax(result))
        query_dict['min'].append(np.amin(result))
        query_dict['count'].append(len(dataset))
        query_dict['std'].append(np.std(result))
        query_dict['emotion'].append(emotion)

        if emotion in data_dict:
            data_dict[emotion] = np.concatenate([data_dict[emotion], result])
        else:
            data_dict[emotion] = result

        t.update()

df = pd.DataFrame(data=query_dict)
for emotion in df.emotion.unique():
    display(df[df.emotion == emotion])

emotion_dict = {
    'emotion': [],
    'mean': [],
    'max': [],
    'min': [],
    'std': [],
    'count': []
}

for emotion, result in data_dict.items():
    emotion_dict['emotion'].append(emotion)
    emotion_dict['mean'].append(np.mean(result))
    emotion_dict['max'].append(np.amax(result))
    emotion_dict['min'].append(np.amin(result))
    emotion_dict['std'].append(np.std(result))
    emotion_dict['count'].append(len(result))

emotion_df = pd.DataFrame(data=emotion_dict)
display(emotion_df)
