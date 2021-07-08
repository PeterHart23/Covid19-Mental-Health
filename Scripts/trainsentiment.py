import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import pandas as pd
import re
from time import time
import nltk
from emoji import demojize
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, GRU
from tensorflow.keras.layers import Dropout, GlobalMaxPooling1D
from tensorflow.keras.layers import Bidirectional, Dense
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split

sentiment140_path = Path('Datasets/sentiment140/sentiment140-edited.csv')
data = pd.read_csv(sentiment140_path)


texts = data.tweet

start = time()
# Lowercasing
texts = texts.str.lower()

# Remove special chars
texts = texts.str.replace(r"(http|@)\S+", "")
texts = texts.apply(demojize)
texts = texts.str.replace(r"::", ": :")
texts = texts.str.replace(r"’", "'")
texts = texts.str.replace(r"[^a-z\':_]", " ")

# Remove repetitions
pattern = re.compile(r"(.)\1{2,}", re.DOTALL)
texts = texts.str.replace(pattern, r"\1")

# Transform short negation form
texts = texts.str.replace(r"(can't|cannot)", 'can not')
texts = texts.str.replace(r"n't", ' not')

# Remove stop words
stopwords = nltk.corpus.stopwords.words('english')
stopwords.remove('not')
stopwords.remove('nor')
stopwords.remove('no')
texts = texts.apply(
    lambda x: ' '.join([word for word in x.split() if word not in stopwords])
)

print("Time to clean up: {:.2f} sec".format(time() - start))

data.tweet = texts

num_words = 10000

tokenizer = Tokenizer(num_words=num_words, lower=True)
tokenizer.fit_on_texts(data.tweet)

file_to_save = Path('Datasets/sentiment140/tokenizer.pickle').resolve()
with file_to_save.open('wb') as file:
    pickle.dump(tokenizer, file)

train = pd.DataFrame(columns=['label', 'tweet'])
validation = pd.DataFrame(columns=['label', 'tweet'])
for label in data.label.unique():
    label_data = data[data.label == label]
    train_data, validation_data = train_test_split(label_data, test_size=0.3)
    train = pd.concat([train, train_data])
    validation = pd.concat([validation, validation_data])


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
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

train_sequences = [text.split() for text in train.tweet]
validation_sequences = [text.split() for text in validation.tweet]
list_tokenized_train = tokenizer.texts_to_sequences(train_sequences)
list_tokenized_validation = tokenizer.texts_to_sequences(validation_sequences)

x_train = pad_sequences(list_tokenized_train, maxlen=input_length)
x_validation = pad_sequences(list_tokenized_validation, maxlen=input_length)
y_train = train.label.replace(4, 1)
y_validation = validation.label.replace(4, 1)


batch_size = 128
epochs = 1

model.fit(
    x_train,
    y=y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_validation, y_validation),
)
model_file = Path('../models/gru_model.h5').resolve()
model.save_weights(model_file.as_posix())
