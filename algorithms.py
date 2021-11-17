"""File containing all algorithms.

Includes:
 - Linear Regression model
 - Long Short-term Memory model
 - ARIMA model

Author: Maria Gospodinova
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
import matplotlib.pyplot as plt
import math
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.metrics import r2_score

from statsmodels.tsa.arima_model import ARIMA
from typing import List


class Algorithms():
    def lstm(self, df: pd.DataFrame) -> List[int]:
        """LSTM model to predict stock price.

        Arguments:
            data {pd.DataFrame} -- stock data

        Returns:
            List[int, int] -- Predicted stock price and RMSE error
        """    """"""
        # 80/20 training/test split
        train = df.iloc[0:int(0.8*len(df)), :]
        test = df.iloc[int(0.8*len(df)):, :]

        # Store as nparray rather than Series object
        train_set = df.iloc[:, 4:5].values

        # Scaling features
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = scaler.fit_transform(train_set)

        # Create a data structure with 1 output and 7 timesteps represeting 7
        # previous days
        X_train = []
        Y_train = []

        for day in range(7, len(train_scaled)):
            X_train.append(train_scaled[day-7:day, 0])
            Y_train.append(train_scaled[day, 0])

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)

        # Reverse arrays
        X_pred = np.array(X_train[-1, 1:])
        X_pred = np.append(X_pred, Y_train[-1])

        # Reshape and add 3rd dimension
        # row x column
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_pred = np.reshape(X_pred, (1, X_pred.shape[0], 1))

        # Initialise RNN
        rnn = Sequential()

        # units - number of neurons in network layer
        # return_sequences - whether to send reccuring memory
        # input_shape - [timesteps] x [(Number of columns) / Features]

        # Layer 1
        rnn.add(LSTM(units=50,
                     return_sequences=True,
                     input_shape=(X_train.shape[1], 1)))
        rnn.add(Dropout(0.1))

        # Layer 2
        rnn.add(LSTM(units=50, return_sequences=True))
        rnn.add(Dropout(0.1))

        # Layer 3
        rnn.add(LSTM(units=50, return_sequences=True))
        rnn.add(Dropout(0.1))

        # Layer 4
        rnn.add(LSTM(units=50))
        rnn.add(Dropout(0.1))

        # O/P layer
        rnn.add(Dense(units=1))

        # Compile RNN
        rnn.compile(optimizer="adam", loss="mean_squared_error")

        # Train RNN
        rnn.fit(X_train, Y_train, epochs=25, batch_size=32)

        # Test RNN
        actual_price = test.iloc[:, 4:5].values

        # To forecast stock price, get closing prices for 7 days before test
        total = pd.concat((train["Close"], test["Close"]), axis=0)
        test_set = total[len(total) - len(test) - 7:].values
        test_set = test_set.reshape(-1, 1)

        # Scaling features
        test_set = scaler.transform(test_set)

        # Testing data structure
        X_test = []

        for day in range(7, len(test_set)):
            X_test.append(test_set[day - 7:day, 0])

        X_test = np.array(X_test)

        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

        # Testing forecase
        forecast_price = rnn.predict(X_test)

        # Reverse feature scaling to get back original prices
        forecast_price = scaler.inverse_transform(forecast_price)

        # Plot figure and export
        lstm_fig = plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(actual_price, label="Actual Price")
        plt.plot(forecast_price, label="Forecast Price")

        plt.legend(loc=4)
        plt.savefig('LSTM.png')
        plt.close(lstm_fig)

        # Calculate root mean squared error (RMSE)
        error_lstm = math.sqrt(mean_squared_error(
            actual_price, forecast_price))

        # Predict future stock price
        predicted_price = rnn.predict(X_pred)

        # Reverse feature scaling to get back original prices
        predicted_price = scaler.inverse_transform(predicted_price)

        lstm_predicted = predicted_price[0, 0]

        print("Tomorrow's Tesla Closing Predicted Price by LSTM is: ",
              lstm_predicted)

        # Algorithm performance metrics
        print("RMSE: ", error_lstm)
        print("MAE:", mean_absolute_error(actual_price, forecast_price))
        print("MAPE:",
              mean_absolute_percentage_error(actual_price, forecast_price))
        print("R2:", r2_score(actual_price, forecast_price))

        return lstm_predicted, error_lstm

    def lin_reg(self, df: pd.DataFrame) -> List[int]:
        """Linear Regression model to predict stock price.

        Arguments:
            df {pd.DataFrame} -- stock data

        Returns:
            List[int, int, int, nparray] -- Mean, predicted stock price,
                                            RMSE error, nparray of predicted
                                            stock prices for the next 7 days
        """
        # Days to forecast
        days_forecast = 7

        # Price after days_forecast
        df["Close after days_forecast"] = df["Close"].shift(-days_forecast)

        # Drop irrelevant data
        new_df = df[["Close", "Close after days_forecast"]]

        # Split data into training and test sets
        X = np.array(new_df.iloc[:-days_forecast, 0: -1])
        Y = np.array(new_df.iloc[:-days_forecast, -1])
        Y = np.reshape(Y, (-1, 1))

        X_to_forecast = np.array(new_df.iloc[-days_forecast:, 0: -1])

        # Train and test to plot graphs and check accuracy
        X_train = X[0:int(0.8*len(df)), :]
        X_test = X[int(0.8*len(df)):, :]
        Y_train = Y[0:int(0.8*len(df)), :]
        Y_test = Y[int(0.8*len(df)):, :]

        # Feature scaling
        std_scale = StandardScaler()
        X_train = std_scale.fit_transform(X_train)
        X_test = std_scale.transform(X_test)
        X_to_forecast = std_scale.transform(X_to_forecast)

        # Train
        clf = LinearRegression(n_jobs=-1)
        clf.fit(X_train, Y_train)

        # Test
        Y_test_pred = clf.predict(X_test)
        Y_test_pred = Y_test_pred*(1.04)

        # Plot
        lin_reg_fig = plt.figure(figsize=(10, 6), dpi=100)
        plt.plot(Y_test, label="Actual Price")
        plt.plot(Y_test_pred, label="Predicted Price")
        plt.legend(loc=4)
        plt.savefig("linear_regression.png")
        plt.close(lin_reg_fig)

        error_lin_reg = math.sqrt(mean_squared_error(Y_test, Y_test_pred))

        # Prediction
        prediction_set = clf.predict(X_to_forecast)
        prediction_set = prediction_set*(1.04)
        mean = prediction_set.mean()
        lin_reg_pred = prediction_set[0, 0]

        print("Tomorrow's Tesla Closing "
              + "Predicted Price by Linear Regression is: ",
              lin_reg_pred)

        # Algorithm performance metrics
        print("RMSE: ", error_lin_reg)
        print("MAE:", mean_absolute_error(Y_test, Y_test_pred))
        print("MAPE:",
              mean_absolute_percentage_error(Y_test, Y_test_pred))
        print("R2:", r2_score(Y_test, Y_test_pred))

        return mean, lin_reg_pred, error_lin_reg, prediction_set

    def arima(self, df: pd.DataFrame) -> List[int]:
        """ARIMA model to predict stock price.

        Arguments:
            df {pd.DataFrame} -- stock data

        Returns:
            List[int, int] -- predicted stock price, RMSE error
        """
        unique_values = df["Ticker"].unique()
        df = df.set_index("Ticker")

        def model(train, test):
            historic = [x for x in train]
            forecast = list()
            for t in range(len(test)):
                model = ARIMA(historic, order=(6, 1, 0))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                forecast.append(yhat[0])
                observation = test[t]
                historic.append(observation)
            return forecast

        for ticker in unique_values[:10]:
            data = (df.loc[ticker, :]).reset_index()
            data["Price"] = data["Close"]
            prc_date = data[["Price", "Date"]]
            prc_date.index = prc_date["Date"].map(lambda t: parse_date(t))
            prc_date["Price"] = prc_date["Price"].map(lambda t: float(t))
            prc_date = prc_date.fillna(prc_date.bfill())
            prc_date = prc_date.drop(["Date"], axis=1)

            arima_fig = plt.figure(figsize=(10, 6), dpi=100)
            plt.plot(prc_date)
            plt.savefig("trends.png")
            plt.close(arima_fig)

            price = prc_date.values
            size = int(len(price) * 0.8)
            train, test = price[0:size], price[size:len(price)]

            # Fit model
            forecast = model(train, test)

            # Plot results
            arima_fig = plt.figure(figsize=(10, 6), dpi=100)
            plt.plot(test, label="Actual Price")
            plt.plot(forecast, label="Predicted Price")
            plt.legend(loc=4)
            plt.savefig("ARIMA.png")
            plt.close(arima_fig)

            arima_predictions = forecast[-2]
            error_arima = math.sqrt(mean_squared_error(test, forecast))

        print("Tomorrow's Tesla Closing "
              + "Predicted Price by ARIMA is: ",
              arima_predictions)

        # Algorithm performance metrics
        print("RMSE: ", error_arima)
        print("MAE:", mean_absolute_error(test, forecast))
        print("MAPE:",
              mean_absolute_percentage_error(test, forecast))
        print("R2:", r2_score(test, forecast))

        return arima_predictions, error_arima


def parse_date(date: datetime.date) -> datetime.date:
    """Parse date to change format.

    Arguments:
        date {datetime.date} -- date

    Returns: datetime.date in YYYY-MM-DD format
    """
    return datetime.strptime(date, "%Y-%m-%d")
