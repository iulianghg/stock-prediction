"""Main application driver.

Author: Maria Gospodinova
"""

import pandas as pd
import credentials
from datetime import datetime
import yfinance as yf
from alpha_vantage.timeseries import TimeSeries

from tweepy import OAuthHandler
import tweets
from telegram import TelegramBot
from typing import List
from algorithms import Algorithms
import database


def get_data(name: str) -> None:
    """Get historical data from Yahoo Finance and store in csv.

    Arguments:
        name {str} -- name of stock to get data for
    """
    end = datetime.now()
    start = datetime(end.year - 2, end.month, end.day)
    data = yf.download(name, start=start, end=end)
    dataframe = pd.DataFrame(data)
    dataframe.to_csv("./" + name + ".csv")

    # If no stock data available on yfinance, dataframe will be empty
    if dataframe.empty:
        # Get data from Alpha Vantage instead
        tser = TimeSeries(key=credentials.AV_KEY, output_format="pandas")
        data = tser.get_daily_adjusted(symbol="NSE:" + name, outputsize="full")

        # Rows from last 2 year - 504
        data = data.head(505).iloc[::-1]
        data = data.reset_index()

        # Only keep required columns, as alphavantage will yield more
        dataframe['Date'] = data['date']
        dataframe['Open'] = data['1. open']
        dataframe['High'] = data['2. high']
        dataframe['Low'] = data['3. low']
        dataframe['Close'] = data['4. close']
        dataframe['Adj Close'] = data['5. adjusted close']
        dataframe['Volume'] = data['6. volume']
        dataframe.to_csv(''+name+'.csv', index=False)


def recommend(avg_pol: int, today_prc: int, mean_prc: int) -> List[str]:
    """Recommend position based on price predicted by algorithms.

    Arguments:
        avg_pol {int} -- Average polarity calculated from sentiment analysis
        today_prc {int} -- Most recent stock price
        mean_prc {int} -- Mean stock price

    Returns:
        List[str, str] -- predicted price direction and recommended action
    """
    if today_prc.iloc[-1]['Close'] < mean_prc:
        if avg_pol > 0:
            print("RISE > BUY")
            price_direction = "rising"
            action = "buy"
        else:
            print("FALL > SELL")
            price_direction = "falling"
            action = "sell"
    else:
        print("FALL > SELL")
        price_direction = "fall"
        action = "sell"
    return price_direction, action


def main() -> None:
    """Get stock data and predict stock price. Stream live Twitter data and
    perform sentiment analysis. Compute a recommendation based on the results
    of ML prediction and sentiment analysis. Send recommendation to Telegram
    bot.

    Returns:
        None
    """
    ticker = "TSLA"

    get_data(ticker)

    df = pd.read_csv("./" + ticker + ".csv")

    # Get latest closing price if market closed
    # Get current price if market open
    today_stock = df.iloc[-1:]

    df = df.dropna()
    ticker_list = []
    for i in range(0, len(df)):
        ticker_list.append(ticker)

    temp_df = pd.DataFrame(ticker_list, columns=["Ticker"])
    temp_df = pd.concat([temp_df, df], axis=1)
    df = temp_df

    Alg = Algorithms()

    arima_predicted, error_arima = Alg.arima(df)
    mean, lr_predicted, error_lr, pred_set = Alg.lin_reg(df)
    lstm_predicted, error_lstm = Alg.lstm(df)

    ticker_map = pd.read_csv("tickers.csv")
    ticker_full = ticker_map[ticker_map["Ticker"] == ticker]
    ticker = ticker_full["Name"].to_list()[0][0:12]

    auth = OAuthHandler(credentials.API_KEY,
                        credentials.API_SECRET_KEY)
    auth.set_access_token(credentials.ACCESS_TOKEN,
                          credentials.ACCESS_TOKEN_SECRET)

    twitter_stream = tweets.StockListener(
        consumer_key=credentials.API_KEY,
        consumer_secret=credentials.API_SECRET_KEY,
        access_token=credentials.ACCESS_TOKEN,
        access_token_secret=credentials.ACCESS_TOKEN_SECRET
    )

    while True:
        # This will keep the stream alive unless we hit Ctrl+C
        try:
            # This is a blocking call
            # (i.e. it will continue until the stream is closed)
            twitter_stream.filter(languages=["en"], track=["Tesla"])
        except KeyboardInterrupt:
            print("Stopped")
            twitter_stream.disconnect()
            break

    # Initialise database object
    db = database.Database()

    # Get average polarity from MySQL database
    db.cursor_execute(database.AVG_POL_QUERY)
    avg_polarity = db.cursor.fetchall()

    # Fetchall method returns results as a tuple, so we need to access the
    # first element of the tuple
    avg_polarity = avg_polarity[0][0]

    # Make a recommendation
    price_direction, action = recommend(avg_polarity, today_stock, mean)

    # Build Telegram message
    message = str(ticker + " price is " + price_direction + ".\n"
                  + "Recommendation: " + action + "!\n")

    # Send to Telegram
    tbot = TelegramBot()
    tbot.send_telegram(message)


if __name__ == "__main__":
    main()
