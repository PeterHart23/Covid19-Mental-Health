import os
import sys
from pathlib import Path
import numpy as np
import pickle
import time
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Conv1D, Dense, concatenate
from tensorflow.keras.models import Model
import pandas as pd
from utils import preprocess
from tensorflow.keras.preprocessing.sequence import pad_sequences
from OhioSort import month, year


tokenizer_path = Path('Datasets/tokenizer.pickle').resolve()
with tokenizer_path.open('rb') as file:
    tokenizer = pickle.load(file)

month = '09'
year = '2020'

# url = f'Datasets/States/Ohio/Ohio{month}_{year}.csv'
url = f'Datasets/Canada/AllCanada.csv'
input_dim = min(tokenizer.num_words, len(tokenizer.word_index) + 1)
num_classes = 4
embedding_dim = 500
input_length = 100
lstm_units = 128
lstm_dropout = 0.1
recurrent_dropout = 0.1
spatial_dropout=0.2
filters=64
kernel_size=3

input_layer = Input(shape=(input_length,))
print("input: ",input_layer)
output_layer = Embedding(
  input_dim=input_dim,
  output_dim=embedding_dim,
  input_shape=(input_length,)
)(input_layer)
print("output: ", output_layer)
output_layer = SpatialDropout1D(spatial_dropout)(output_layer)

output_layer = Bidirectional(
LSTM(lstm_units, return_sequences=True,
     dropout=lstm_dropout, recurrent_dropout=recurrent_dropout)
)(output_layer)
output_layer = Conv1D(filters, kernel_size=kernel_size, padding='valid',
                    kernel_initializer='glorot_uniform')(output_layer)

avg_pool = GlobalAveragePooling1D()(output_layer)
max_pool = GlobalMaxPooling1D()(output_layer)
output_layer = concatenate([avg_pool, max_pool])

output_layer = Dense(num_classes, activation='softmax')(output_layer)

model = Model(input_layer, output_layer)
model_weights_path = Path('../models/model_weights.h5').resolve()
model.load_weights(model_weights_path.as_posix())


data_path = Path(url).resolve()
data = pd.read_csv(data_path)
# data.insert(9,"break","")
data.insert(10,"anger","")
data.insert(11,"fear","")
data.insert(12,"joy","")
data.insert(13,"sadness","")
data.insert(14,"emotion","")
data.head()
#

start_time=time.time()


encoder_path = Path('../models/encoder.pickle').resolve()
with encoder_path.open('rb') as file:
    encoder = pickle.load(file)

cleaned_data = preprocess(data.TweetMessage)
sequences = [text.split() for text in cleaned_data]
list_tokenized = tokenizer.texts_to_sequences(sequences)
x_data = pad_sequences(list_tokenized, maxlen=100)
# print(time.time()-start_time)

y_pred = model.predict(x_data)
data.anger = y_pred[:,0]
data.fear = y_pred[:,1]
data.joy = y_pred[:,2]
data.sadness = y_pred[:,3]

emotionswitch = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "sadness",
}
# print(time.time()-start_time)
for i in range(len(y_pred)):
    data.emotion[i] = emotionswitch.get(np.argmax(y_pred[i]))

    # data.emotion[i](np.argmax(y_pred[i]))

# max_index, max_value = max(enumerate(y_pred[1]), key=operator.itemgetter(1))

# for i in range(len(y_pred)):
#     max = 0
#     count = 0
#     for j in range(len(y_pred[0])):
#         if y_pred[i][j] > max:
#             max = y_pred[i][j]
#             count = j
#     data.emotion[i] = emotionswitch.get(count)
data.to_csv(url, sep=',', encoding='utf-8',index=False)

for index, value in enumerate(np.sum(y_pred, axis=0) / len(y_pred)):
    print(encoder.classes_[index] + ": " + str(value))

# y_pred_argmax = y_pred.argmax(axis=1)
# data_len = len(y_pred_argmax)
# for index, value in enumerate(np.unique(y_pred_argmax)):
#     print(encoder.classes_[index] + ": " + str(len(y_pred_argmax[y_pred_argmax == value]) / data_len))

y_pred[5:10].argmax(axis=1)
# print(time.time()-start_time)
