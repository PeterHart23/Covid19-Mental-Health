import os
import re
import json
import pandas as pd
from pathlib import Path
from emoji import demojize, emojize

relations_path = Path('query_relations.json')
with relations_path.open('r') as file:
    relations = json.load(file)

emotion = 'anger'
queries = [key for key, value in relations.items() if value == emotion]

files_dir = Path('Datasets/tweepy').resolve()
data = []
for filename in os.listdir(files_dir):
    file_query = re.findall(r'(#[^.]+|:.+:)', filename)[0]
    if file_query in queries:
        data += [pd.read_csv(os.path.join(files_dir, filename))]

data = pd.concat(data)
data_emojis = data.text.apply(lambda x: re.findall(r':[a-z_]+:', demojize(x)))

emoji_dict = {}
for i, emojis in data_emojis.iteritems():
    for emoji in emojis:
        if emoji in emoji_dict:
            emoji_dict[emoji] += 1
        else:
            emoji_dict[emoji] = 1

data_hashtags = data.text.apply(lambda x: re.findall(r'#\S+', x))
hashtag_dict = {}
for i, hashtags in data_hashtags.iteritems():
    for hashtag in hashtags:
        if hashtag in hashtag_dict:
            hashtag_dict[hashtag] += 1
        else:
            hashtag_dict[hashtag] = 1

for emoji, count in sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True):
    print(emojize(emoji) + '(' + emoji + '): ' + str(count))

for hashtag, count in sorted(hashtag_dict.items(), key=lambda x: x[1], reverse=True):
    print(hashtag + ': ' + str(count))
