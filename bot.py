import tweepy

consumer_key = "xxx"
consumer_secret = "xxx"
bearer = "xxx"

access = "xxx"
access_secret = "xxx"

client = tweepy.Client(bearer_token=bearer, consumer_key= consumer_key, consumer_secret=consumer_secret, access_token=access, access_token_secret=access_secret)
client.create_tweet(text = "joe")