from pathlib import Path
import os
from tweepy import OAuthHandler, API, TweepError
from emoji import emojize
import pandas as pd

#
# env_path = Path('../.env').resolve()
# load_dotenv(dotenv_path=env_path)



consumer_key = "LmVv6g9pfcFzAXIV0eeyov47J"
consumer_secret = "micZqLIRsHB1Ms9jMTvfD3guTuhHAqdrQ46n7E8vI3roV6LksT"
access_token = "918068377-weue49sTSSCOyA8zb1DUwSCxT0Sy4Xs8lbyYGfnX"
access_token_secret = "N6kDvcYkp2mEsxVYsF6rZw9rYDhv2i7SHBA6Wk0pRyjAx"


auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = API(auth)
print('Successfully connected to the Twitter API.')

query = '#depressed'
max_requests = 180


q = emojize(query) + ' -filter:retweets'
searched_tweets = []
last_id = -1
request_count = 0
while request_count < max_requests:
    try:
        new_tweets = api.search(q=q,
                                lang='en',
                                count=100,
                                max_id=str(last_id - 1),
                                tweet_mode='extended')
        if not new_tweets:
            break
        searched_tweets.extend(new_tweets)
        last_id = new_tweets[-1].id
        request_count += 1
    except TweepError as e:
        print(e)
        break

data = []
for tweet in searched_tweets:
    data.append([tweet.id, tweet.created_at, tweet.user.screen_name, tweet.full_text])
df = pd.DataFrame(data=data, columns=['id', 'date', 'user', 'text'])
print(str(len(data)) + ' ' + query + ' tweets')
df.head()

PATH = Path('Datasets/TestData').resolve()
filename = query + '.csv'
df.to_csv(os.path.join(PATH, filename), index=None)
print('Saved under: "' + PATH.as_posix() + '"')
