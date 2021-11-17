"""File containing class for streaming from Twitter.

Author: Maria Gospodinova
"""

import tweepy
from textblob import TextBlob
import database
import re
from collections import defaultdict
from typing import List


class StockListener(tweepy.Stream):
    """Polymorphed tweepy.Stream to connect to Twitter API and extract data for
    sentiment analysis.
    """

    def on_status(self, status: tweepy) -> List:
        """Method called when a Tweet status is received.

        Arguments:
            status {tweepy} - Tweet status

        Returns:
            List {Dict, int} - Dictionary {tweet, polarity} and polarity sum
        """
        # Do not take into account retweeted data
        if status.retweeted or "RT" in status.text:
            return True

        # Tweet data
        id_str = status.id_str
        created_at = status.created_at
        text = self.remove_emojis(self.clean_tweet(status.text))
        blob = TextBlob(text)

        tweets_dict = defaultdict(list)

        polarity = 0
        polarity_sum = 0
        positive_tweets = 0
        negative_tweets = 0
        neutral_tweets = 0

        for sentence in blob.sentences:
            polarity += sentence.sentiment.polarity

            if polarity > 0:
                positive_tweets = positive_tweets + 1
            elif polarity < -0:
                negative_tweets = negative_tweets + 1
            else:
                neutral_tweets = neutral_tweets + 1

            polarity_sum += sentence.sentiment.polarity

        tweets_dict.update({polarity: text})

        if len(tweets_dict) != 0:
            polarity_sum = polarity_sum / len(tweets_dict)

        # Instantiate database object
        db = database.Database()

        # Create database if it does not exist
        db.cursor_execute(database.CREATE_DB_QUERY)

        # Create table if it does not exist
        db.cursor_execute(database.CREATE_TABLE_QUERY)

        # Load attributes in a list to prepare for upload to database
        database_columns = [id_str, created_at, text, polarity]

        db.insert_tweet(database_columns)
        db.commit()

        print("Tweet collected.")

        return tweets_dict, polarity_sum

    def on_error(self, status: tweepy) -> None:
        """Method called on stream error.

        Arguments:
            status {tweepy} - Error status

        Returns:
            None
        """
        print(status)

    def error_handle(self, status_code: int) -> bool:
        """Stop scraping data if Twitter API rate threshold is exceeded.

        Arguments:
            status_code {int} -- HTTP status code

        Returns:
            bool -- True if stream rate limit exceeded
        """
        if status_code == 420:
            print("Twitter API rate threshold exceeded. Disconnected stream.")
            return False    # Disconnect data stream

    def clean_tweet(self, tweet: tweepy) -> str:
        """Clean tweet text by removing URLs, hashtags, and usernames.

        Arguments:
            tweet {tweepy} -- tweet text to clean

        Returns:
            str -- cleaned tweet text
        """
        return ' '.join(re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) \
                                |(\w+:\/\/\S+)", " ", tweet).split())

    def remove_emojis(self, text: str) -> str:
        """Remove emojis by encoding to ASCII.
        Ignore characters outside the ASCII table, and decode back to UTF-8.

        Arguments:
            text {str} -- Tweet text to remove emojis from

        Returns:
            str -- Tweet text without emojis
        """
        # Only execute if non-empty string is passed
        if text:
            return text.encode('ascii', 'ignore').decode('ascii')
        else:
            print("No text.")
            return None
