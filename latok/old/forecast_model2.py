# forecast_model2.py

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.vector_ar.var_model import VAR
from arch import arch_model
from sklearn.metrics import mean_squared_error
from math import sqrt
from datafetchSrc import OHLCVDataFetcher
import time
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

    def train_arima_model(self, order=(1,1,1)):
        if not self.prepare_data():
            return False

        self.model = ARIMA(self.train_data, order=order)
        self.model_fit = self.model.fit()
        return True

    def train_sarima_model(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        if not self.prepare_data():
            return False

        self.model = SARIMAX(self.train_data, order=order, seasonal_order=seasonal_order)
        self.model_fit = self.model.fit()
        return True

    def train_garch_model(self, p=1, q=1):
        if not self.prepare_data():
            return False

        returns = pd.Series(self.train_data).pct_change().dropna()
        returns = returns * 100

        self.model = arch_model(returns, vol='GARCH', p=p, q=q)
        self.model_fit = self.model.fit()
        return True


    def train_var_model(self, lags=5):
        if self.data is None:
            print("No data available. Please fetch data first.")
            return False
        self.train_data = self.data[['open', 'high', 'low', 'close', 'volume']].values
        self.model = VAR(self.train_data)
        self.model_fit = self.model.fit(lags)
        return True

    def make_forecast(self, steps=5, model_type='arima'):
        if self.model_fit is None:
            print(f"{model_type.upper()} model not trained. Please train the model first.")
            return None
        if model_type == 'arima':
            forecast = self.model_fit.forecast(steps=steps)
        elif model_type == 'sarima':
            forecast = self.model_fit.forecast(steps=steps)
        elif model_type == 'garch':
            forecast = self.model_fit.forecast(horizon=steps)
            return forecast.variance.values[-1]  # Return volatility forecast
        elif model_type == 'var':
            forecast = self.model_fit.forecast(self.train_data[-self.model_fit.k_ar:], steps=steps)
        else:
            print("Invalid model type. Choose from 'arima', 'sarima', 'garch', or 'var'.")
            return None

        return forecast


    def evaluate_model(self, model_type='arima'):
        if self.model_fit is None:
            print(f"{model_type.upper()} model not trained. Please train the model first.")
            return None

        # Use the last 20% of the data as a test set
        test_size = int(len(self.train_data) * 0.2)
        train, test = self.train_data[:-test_size], self.train_data[-test_size:]

        if model_type in ['arima', 'sarima']:
            predictions = self.model_fit.forecast(steps=len(test))
        elif model_type == 'garch':
            returns = pd.Series(self.train_data).pct_change().dropna() * 100
            predictions = self.model_fit.forecast(horizon=len(test)).variance.values[-1]
            test = returns[-len(test):]**2  # Compare with squared returns
        elif model_type == 'var':
            predictions = self.model_fit.forecast(train[-self.model_fit.k_ar:], steps=len(test))
            predictions = predictions[:, 3]  # Assuming we're interested in 'close' price predictions
            test = test[:, 3]  # 'close' price from test data

        rmse = sqrt(mean_squared_error(test, predictions))
        return rmse

def main():
    symbol = "LUNC/USDT"
    daily = "1d"
    weekly = "1w"
    monthly = "1M"
    LIGHT_CYAN = "\033[1;34m"
    LIGHT_PURPLE = "\033[1;35m"
    LIGHT_GREEN = "\033[1;32m"
    RESET = "\033[0m"

    daily_model = ForecastModel(symbol, daily)
    weekly_model = ForecastModel(symbol, weekly)
    monthly_model = ForecastModel(symbol, monthly)

#------------------------------------------------------------
    time.sleep(15)

    if daily_model.fetch_data():
        print("Data fetched successfully.")

        # ARIMA
        if daily_model.train_arima_model():
            print("ARIMA Model trained successfully.")
            arima_forecast = daily_model.make_forecast(steps=5, model_type='arima')
            print(f"\n5-day ARIMA forecast: {LIGHT_CYAN}{arima_forecast}{RESET}")
            arima_rmse = daily_model.evaluate_model(model_type='arima')
            print(f"ARIMA Model RMSE: {LIGHT_CYAN}{arima_rmse:.8f}\n{RESET}")

        # SARIMA
        if daily_model.train_sarima_model():
            print("\nSARIMA Model trained successfully.")
            sarima_forecast = daily_model.make_forecast(steps=5, model_type='sarima')
            print(f"5-day SARIMA forecast: {LIGHT_CYAN}{sarima_forecast}{RESET}")
            sarima_rmse = daily_model.evaluate_model(model_type='sarima')
            print(f"SARIMA Model RMSE: {LIGHT_CYAN}{sarima_rmse:.8f}\n{RESET}")

        # GARCH
        if daily_model.train_garch_model():
            print("\nGARCH Model trained successfully.")
            garch_forecast = daily_model.make_forecast(steps=5, model_type='garch')
            print(f"5-day GARCH volatility forecast: {LIGHT_CYAN}{garch_forecast}{RESET}")
            garch_rmse = daily_model.evaluate_model(model_type='garch')
            print(f"GARCH Model RMSE: {LIGHT_CYAN}{garch_rmse:.8f}\n{RESET}")

        # VAR
        if daily_model.train_var_model():
            print("\nVAR Model trained successfully.")
            var_forecast = daily_model.make_forecast(steps=5, model_type='var')
            print(f"5-day VAR forecast:\n{LIGHT_CYAN}{var_forecast}{RESET}")
            var_rmse = daily_model.evaluate_model(model_type='var')
            print(f"VAR Model RMSE: {LIGHT_CYAN}{var_rmse:.8f}\n{RESET}")

    else:
        print("Failed to fetch data.")

# #----------------------------------------------------------------
    if weekly_model.fetch_data():
        print("Data fetched successfully.")

        # ARIMA
        if weekly_model.train_arima_model():
            print("ARIMA Model trained successfully.")
            arima_forecast = weekly_model.make_forecast(steps=5, model_type='arima')
            print(f"\n5-week ARIMA forecast: {LIGHT_PURPLE}{arima_forecast}{RESET}")
            arima_rmse = weekly_model.evaluate_model(model_type='arima')
            print(f"ARIMA Model RMSE: {LIGHT_PURPLE}{arima_rmse:.8f}{RESET}")

        # SARIMA
        if weekly_model.train_sarima_model():
            print("\nSARIMA Model trained successfully.")
            sarima_forecast = weekly_model.make_forecast(steps=5, model_type='sarima')
            print(f"5-week SARIMA forecast:{LIGHT_PURPLE} {sarima_forecast}{RESET}")
            sarima_rmse = weekly_model.evaluate_model(model_type='sarima')
            print(f"SARIMA Model RMSE:{LIGHT_PURPLE} {sarima_rmse:.8f}{RESET}")

        # GARCH
        if weekly_model.train_garch_model():
            print("\nGARCH Model trained successfully.")
            garch_forecast = weekly_model.make_forecast(steps=5, model_type='garch')
            print(f"5-week GARCH volatility forecast:{LIGHT_PURPLE} {garch_forecast}{RESET}")
            garch_rmse = weekly_model.evaluate_model(model_type='garch')
            print(f"GARCH Model RMSE:{LIGHT_PURPLE} {garch_rmse:.8f}{RESET}")

        # VAR
        if weekly_model.train_var_model():
            print("\nVAR Model trained successfully.")
            var_forecast = weekly_model.make_forecast(steps=5, model_type='var')
            print(f"5-week VAR forecast:\n{LIGHT_PURPLE}{var_forecast}{RESET}")
            var_rmse = weekly_model.evaluate_model(model_type='var')
            print(f"VAR Model RMSE:{LIGHT_PURPLE} {var_rmse:.8f}{RESET}")

    else:
        print("Failed to fetch data.")

# #------------------------------------------------------------
    if monthly_model.fetch_data():
        print("Data fetched successfully.")

        # ARIMA
        if monthly_model.train_arima_model():
            print("ARIMA Model trained successfully.")
            arima_forecast = monthly_model.make_forecast(steps=5, model_type='arima')
            print(f"\n5-month ARIMA forecast: {LIGHT_GREEN}{arima_forecast}{RESET}")
            arima_rmse = monthly_model.evaluate_model(model_type='arima')
            print(f"ARIMA Model RMSE: {LIGHT_GREEN}{arima_rmse:.8f}{RESET}")

        # SARIMA
        if monthly_model.train_sarima_model():
            print("\nSARIMA Model trained successfully.")
            sarima_forecast = monthly_model.make_forecast(steps=5, model_type='sarima')
            print(f"5-month SARIMA forecast: {LIGHT_GREEN}{sarima_forecast}{RESET}")
            sarima_rmse = monthly_model.evaluate_model(model_type='sarima')
            print(f"SARIMA Model RMSE: {LIGHT_GREEN}{sarima_rmse:.8f}{RESET}")

        # GARCH
        if monthly_model.train_garch_model():
            print("\nGARCH Model trained successfully.")
            garch_forecast = monthly_model.make_forecast(steps=5, model_type='garch')
            print(f"5-month GARCH volatility forecast: {LIGHT_GREEN}{garch_forecast}{RESET}")
            garch_rmse = monthly_model.evaluate_model(model_type='garch')
            print(f"GARCH Model RMSE: {LIGHT_GREEN}{garch_rmse:.8f}{RESET}")

        # VAR
        if monthly_model.train_var_model():
            print("\nVAR Model trained successfully.")
            var_forecast = monthly_model.make_forecast(steps=5, model_type='var')
            print(f"5-month VAR forecast:\n{LIGHT_GREEN}{var_forecast}{RESET}")
            var_rmse = monthly_model.evaluate_model(model_type='var')
            print(f"VAR Model RMSE: {LIGHT_GREEN}{var_rmse:.8f}{RESET}")

    else:
        print("Failed to fetch data.")

# #------------------------------------------------------------

if __name__ == "__main__":
    main()
