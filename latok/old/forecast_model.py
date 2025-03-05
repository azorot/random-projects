# forecast_model.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt
from datafetchSrc import OHLCVDataFetcher

class ForecastModel:
    def __init__(self, symbol, resolution):
        self.fetcher = OHLCVDataFetcher()
        self.symbol = symbol
        self.resolution = resolution
        self.data = None
        self.model = None

    def fetch_data(self):
        self.data = self.fetcher.get_ohlcv_dataframe(self.symbol, self.resolution)
        if self.data is not None:
            self.data.set_index('timestamp', inplace=True)
            self.data.sort_index(inplace=True)
        return self.data is not None

    def prepare_data(self):
        if self.data is None:
            print("No data available. Please fetch data first.")
            return False

        # Use 'close' prices for this example
        self.train_data = self.data['close'].values
        return True

    def train_model(self, order=(1,1,1)):
        if not self.prepare_data():
            return False

        self.model = ARIMA(self.train_data, order=order)
        self.model_fit = self.model.fit()
        return True

    def make_forecast(self, steps=5):
        if self.model_fit is None:
            print("Model not trained. Please train the model first.")
            return None

        forecast = self.model_fit.forecast(steps=steps)
        return forecast

    def evaluate_model(self):
        if self.model_fit is None:
            print("Model not trained. Please train the model first.")
            return None

        # Use the last 20% of the data as a test set
        test_size = int(len(self.train_data) * 0.2)
        train, test = self.train_data[:-test_size], self.train_data[-test_size:]

        # Fit the model on the training data
        history = [x for x in train]
        predictions = []
        for t in range(len(test)):
            model = ARIMA(history, order=(1,1,1))
            model_fit = model.fit()
            output = model_fit.forecast()
            yhat = output[0]
            predictions.append(yhat)
            obs = test[t]
            history.append(obs)

        # Calculate RMSE
        rmse = sqrt(mean_squared_error(test, predictions))
        return rmse

def main():
    symbol = "LUNC/USDT"
    daily = "1d"
    weekly = "1w"
    monthly = "1M"


    daily_model = ForecastModel(symbol, daily)
    weekly_model = ForecastModel(symbol, weekly)
    monthly_model = ForecastModel(symbol, monthly)
#------------------------------------------------------------
    if daily_model.fetch_data():
        print(f"Data fetched successfully.(daily)")

        if daily_model.train_model():
            print(f"daily Model trained successfully.(daily)")

            daily_forecast = daily_model.make_forecast(steps=5)
            print(f"\n5-day forecast: {daily_forecast}")

            daily_rmse = daily_model.evaluate_model()
            print(f"daily Model RMSE: {daily_rmse:.8f}\n")
        else:
            print("Failed to train the model.(daily)")
    else:
        print("Failed to fetch data.(daily)")
#----------------------------------------------------------------
    if weekly_model.fetch_data():
        print("Data fetched successfully.(weekly)")

        if weekly_model.train_model():
            print("weekly Model trained successfully.(weekly)")

            weekly_forecast = weekly_model.make_forecast(steps=5)
            print(f"\n5-Week forecast: {weekly_forecast}")

            weekly_rmse = weekly_model.evaluate_model()
            print(f"weekly Model RMSE: {weekly_rmse:.8f}\n")
        else:
            print("Failed to train the model.(weekly)")
    else:
        print("Failed to fetch data.(weekly)")
#------------------------------------------------------------
    if monthly_model.fetch_data():
        print("Data fetched successfully.(monthly)")

        if monthly_model.train_model():
            print("monthly Model trained successfully.(monthly)")

            monthly_forecast = monthly_model.make_forecast(steps=5)
            print(f"\n5-Month forecast: {monthly_forecast}")

            monthly_rmse = monthly_model.evaluate_model()
            print(f"monthly Model RMSE: {monthly_rmse:.4f}")
        else:
            print("Failed to train the model.(monthly)")
    else:
        print("Failed to fetch data.(monthly)")
#------------------------------------------------------------

if __name__ == "__main__":
    main()
