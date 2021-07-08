import os
import re
import pandas as pd
from tqdm import tqdm
import sys
from pathlib import Path
import pickle
import numpy as np
from emoji import demojize
from tensorflow.keras.preprocessing.sequence import pad_sequences
from utils import preprocess
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Embedding, GRU
from tensorflow.keras.layers import Dropout, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Dense
from tensorflow.keras.models import Sequential

sys.path.append(Path(os.path.join(os.path.abspath(''), '../')).resolve().as_posix())
import json
from pathlib import Path

relations_path = Path('query_relations.json').resolve()
with relations_path.open('rb') as file:
    relations = json.load(file)

tokenizer_path = Path('Datasets/sentiment140/tokenizer.pickle').resolve()
with tokenizer_path.open('rb') as file:
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

print(model.summary())
weights_path = Path('../models/gru_model.h5').resolve()
model.load_weights(weights_path.as_posix())

files_dir = Path('Datasets/tweepy').resolve()
emotion_data_dict = {}
filenames = os.listdir(files_dir)
print(filenames)
# del filenames[-1]
with tqdm(total=len(filenames)) as t:
    for filename in filenames:
        if filename is not '.DS_Store':
            print(filename)
            query = re.findall(r'(#[^.]+|:.+:)', filename)[0]
            emotion = relations[query]

            file_data = pd.read_csv(os.path.join(files_dir, filename))
            dict_data = emotion_data_dict[emotion] if emotion in emotion_data_dict else None
            emotion_data_dict[emotion] = pd.concat([dict_data, file_data])
            t.update()

def get_score_range(mean):
  if mean < 0.5:
    return (0.0, mean)
  return (mean, 1.0)



result_data = []

messages = []
with tqdm(total=len(emotion_data_dict.items())) as t:
    for emotion, dataset in emotion_data_dict.items():
        t.set_description('Processing "' + emotion + '" data')
        cleaned_texts = preprocess(dataset.text, quiet=True)
        predict_sequences = [text.split() for text in cleaned_texts]
        list_tokenized_predict = tokenizer.texts_to_sequences(predict_sequences)
        x_predict = pad_sequences(list_tokenized_predict, maxlen=100)

        result = model.predict(x_predict)
        mean = np.mean(result)
        std = np.std(result)
        low, high = get_score_range(mean)
        messages.append(emotion.capitalize() + ": Score Range: {:4f} - {:4f}".format(low, high))
        dataset = dataset[np.all([(result >= low), (result <= high)], axis=0)]
        dataset.insert(0, 'label', emotion)

        result_data = result_data + [dataset]
        t.update()

for message in messages:
    print(message)

if len(result_data) > 0:
    result_data = pd.concat(result_data)

    path = Path('Datasets/testdata.csv').resolve()
    result_data.to_csv(path, index=None)

    print('Files saved under "' + path.as_posix() + '"')
