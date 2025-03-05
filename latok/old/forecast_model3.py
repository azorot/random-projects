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
from numpy import log1p, sqrt, std, subtract, mean, diff, abs, log
YELLOW = "\033[1;33m"
RED = "\033[0;31m"
LIGHT_CYAN = "\033[1;34m"
LIGHT_PURPLE = "\033[1;35m"
PURPLE = "\033[0;35m"
LIGHT_GREEN = "\033[1;32m"
END= "\033[0m"
GREEN = "\033[0;32m"
BLUE = "\033[0;34m"
forwardH = 8760
forwardD = 365
forwardW = 52
forwardM = 1

class ForecastModel:
    def __init__(self, symbol, resolution):
        self.fetcher = OHLCVDataFetcher()
        self.symbol = symbol
        self.resolution = resolution
        self.data = None
        self.model = None

    def fetch_data(self):
        print(f"{YELLOW}Attempting to fetch data...{END}")
        self.data = self.fetcher.get_ohlcv_dataframe(self.symbol, self.resolution)
        if self.data is not None:
            self.data.set_index('timestamp', inplace=True)
            self.data.sort_index(inplace=True)
            print(f"{LIGHT_GREEN}---Successfully fetched data!{END}")
        return self.data is not None

    def prepare_data(self):
        print(f"{YELLOW}Attempting to prepare data...{END}")
        if self.data is None:
            print(f"{RED}No data available. Please fetch data first.{END}")
            return False
        print(f"{LIGHT_GREEN}---Successfully prepared the data!{END}")
        self.train_data = self.data['close'].astype(float).values
        return True
#---------------------------------------------------------------------------------------
    def calculate_hurst_exponent(self, lags=20):
        """Calculate the Hurst exponent of the time series."""
        ts = self.train_data
        lags = range(2, lags)
        tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
        poly = np.polyfit(np.log1p(lags), np.log1p(tau), 1)
        return poly[0] * 2.0

    def calculate_fractal_dimension(self, max_lag=100):
        """Calculate the fractal dimension of the time series."""
        ts = self.train_data
        lags = range(1, max_lag)
        var = [std(subtract(ts[lag:], ts[:-lag])) for lag in lags]
        return 2 - (log1p(var[-1]) - log1p(var[0])) / (log1p(lags[-1]) - log1p(lags[0])) / 2

    def estimate_lyapunov_exponent(self, lag=1, min_points=30, epsilon=1e-10):
        """Estimate the Lyapunov exponent of the time series."""
        ts = self.train_data
        n = len(ts)
        divergence = []
        for i in range(n - min_points - lag):
            d0 = subtract(ts[i + 1:i + min_points + 1], ts[i]) + epsilon
            # print(d0)
            d1 = subtract(ts[i + lag + 1:i + lag + min_points + 1], ts[i + lag]) + epsilon
            # print(d1)
            divergence.append(log1p(abs(d1 / d0 - 1)))
        return mean(divergence) / lag

    def analyze_chaos_dynamics(self):
        print(f"{YELLOW}Attempting to analyze chaos dynamics...{END}")
        if not self.prepare_data():
            print(f"{RED}No data available. Please fetch data first.{END}")
            return False
        print(f"{LIGHT_GREEN}---Successfully analyzed chaos dynamics!{END}")
        hurst = self.calculate_hurst_exponent()
        fractal_dim = self.calculate_fractal_dimension()
        lyapunov = self.estimate_lyapunov_exponent()

        print(f"\n")
        print(f"Hurst Exponent: {hurst:.4f}")
        print(f"Fractal Dimension: {fractal_dim:.4f}")
        print(f"Lyapunov Exponent: {lyapunov:.4f}")

        print(f"\n")

        return True
#---------------------------------------------------------------------------------------
    def train_arima_model(self, order=(1,1,1)):
        print(f"{YELLOW}Attempting to train ARIMA model...{END}")
        if not self.prepare_data():
            print(f"{RED}No data available. Please fetch data first.{END}")
            return False

        self.model = ARIMA(self.train_data, order=order)
        print(f"{LIGHT_GREEN}---Successfully trained ARIMA model!{END}")
        print(f"{YELLOW}Attempting to fit ARIMA model...{END}")
        self.model_fit = self.model.fit()
        print(f"{LIGHT_GREEN}---Successfully fitted ARIMA model!{END}")
        return True

    def train_sarima_model(self, order=(1,1,1), seasonal_order=(1,1,1,12)):
        print(f"{YELLOW}Attempting to train SARIMA model...{END}")
        if not self.prepare_data():
            print(f"{RED}No data available. Please fetch data first.{END}")
            return False

        self.model = SARIMAX(self.train_data, order=order, seasonal_order=seasonal_order)
        print(f"{LIGHT_GREEN}---Successfully trained SARIMA model!{END}")
        print(f"{YELLOW}Attempting to fit SARIMA model...{END}")
        self.model_fit = self.model.fit(disp=0)
        print(f"{LIGHT_GREEN}---Successfully fitted SARIMA model!{END}")
        return True

    def train_garch_model(self, p=1, q=1):
        print(f"{YELLOW}Attempting to train GARCH model...{END}")
        if not self.prepare_data():
            print(f"{RED}No data available. Please fetch data first.{END}")
            return False

        returns = pd.Series(self.train_data).pct_change().dropna()
        returns = returns * 100

        self.model = arch_model(returns, vol='GARCH', p=p, q=q)
        print(f"{LIGHT_GREEN}---Successfully trained GARCH model!{END}")
        print(f"{YELLOW}Attempting to fit GARCH model...{END}")
        self.model_fit = self.model.fit(disp=0)
        print(f"{LIGHT_GREEN}---Successfully fitted GARCH model!{END}")
        return True


    def train_var_model(self, lags=5):
        print(f"{YELLOW}Attempting to train VAR model...{END}")
        if self.data is None:
            print(f"{RED}No data available. Please fetch data first.{END}")
            return False
        self.train_data = self.data[['open', 'high', 'low', 'close', 'volume']].values
        self.model = VAR(self.train_data)
        print(f"{LIGHT_GREEN}---Successfully trained VAR model!{END}")
        print(f"{YELLOW}Attempting to fit VAR model...{END}")
        self.model_fit = self.model.fit(lags)
        print(f"{LIGHT_GREEN}---Successfully fitted VAR model!{END}")
        return True

    def make_forecast(self, steps=5, model_type='arima'):
        if self.model_fit is None:
            print(f"{model_type.upper()} model not trained. Please train the model first!")
            return None

        try:
            if model_type == 'arima':
                print(f"{BLUE}Attempting ARIMA forecast...{END}")
                forecast = self.model_fit.forecast(steps=steps)
            elif model_type == 'sarima':
                print(f"{BLUE}Attempting SARIMA forecast...{END}")
                forecast = self.model_fit.forecast(steps=steps)
            elif model_type == 'garch':
                print(f"{BLUE}Attempting GARCH forecast...{END}")
                forecast = self.model_fit.forecast(horizon=steps)  # Ensure this is correct
                return forecast.variance.values[-1]  # Return volatility forecast
            elif model_type == 'var':
                if len(self.train_data) < self.model_fit.k_ar:
                    print("Not enough data points for VAR forecasting.")
                    return None

                last_values = self.train_data[-self.model_fit.k_ar:]  # Get last k_ar values
                print(f"{BLUE}Attempting VAR forecast...{END}")
                forecast = self.model_fit.forecast(y=last_values, steps=steps)
            else:
                print("Invalid model type. Choose from 'arima', 'sarima', 'garch', or 'var'.")
                return None

        except Exception as e:
            print(f"{RED}Error during forecasting with {model_type}: {e}{END}")
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
    hourly = "60"
    daily = "1d"
    weekly = "1w"
    monthly = "1M"

    hourly_model = ForecastModel(symbol, hourly)
    daily_model = ForecastModel(symbol, daily)
    weekly_model = ForecastModel(symbol, weekly)
    monthly_model = ForecastModel(symbol, monthly)

#------------------------------------------------------------
    time.sleep(1)
    if hourly_model.fetch_data():
        # print("Data fetched successfully.")
        hourly_model.analyze_chaos_dynamics()
        # ARIMA
        if hourly_model.train_arima_model():
            # print("ARIMA Model trained successfully.")
            arima_forecast = hourly_model.make_forecast(steps=forwardH, model_type='arima')
            print(f"\n{forwardH}-hour ARIMA forecast: \n{LIGHT_CYAN}{arima_forecast}{END}")
            arima_rmse = hourly_model.evaluate_model(model_type='arima')
            print(f"ARIMA Model RMSE: {LIGHT_CYAN}{arima_rmse:.8f}\n{END}")
            amean = arima_forecast.mean()
            print(f"ARIMA mean: {amean:.8f}")
            sub = amean-  arima_rmse
            add = arima_rmse + amean
            print(f"adder: {add:.8f} | subtract: {sub:.8f}")
        # SARIMA
        if hourly_model.train_sarima_model():
            # print("\nSARIMA Model trained successfully.")
            sarima_forecast = hourly_model.make_forecast(steps=forwardH, model_type='sarima')
            print(f'{forwardH}-hour SARIMA forecast: \n{LIGHT_CYAN}{sarima_forecast}{END}')
            sarima_rmse = hourly_model.evaluate_model(model_type='sarima')
            print(f"SARIMA Model RMSE: {LIGHT_CYAN}{sarima_rmse:.8f}\n{END}")
            smean = sarima_forecast.mean()
            print(f"SARIMA mean: {smean:.8f}")
        # GARCH
        if hourly_model.train_garch_model():
            # print("\nGARCH Model trained successfully.")
            garch_forecast = hourly_model.make_forecast(steps=forwardH, model_type='garch')
            print(f"{forwardH}-hour GARCH volatility forecast: \n{LIGHT_CYAN}{garch_forecast}{END}")
            garch_rmse = hourly_model.evaluate_model(model_type='garch')
            print(f"GARCH Model RMSE: {LIGHT_CYAN}{garch_rmse:.8f}\n{END}")
            gmean = garch_forecast.mean()
            print(f"GARCH mean: {gmean:.8f}")
        # VAR
        if hourly_model.train_var_model():
            # print("\nVAR Model trained successfully.")
            var_forecast = hourly_model.make_forecast(steps=forwardH, model_type='var')
            print(f"{forwardH}-hour VAR forecast: \n{LIGHT_CYAN}{var_forecast}{END}")
            var_rmse = hourly_model.evaluate_model(model_type='var')
            print(f"VAR Model RMSE: {LIGHT_CYAN}{var_rmse:.8f}\n{END}")
            vmean = var_forecast.mean()
            print(f"VAR mean: {vmean:.8f}")
            print(f"-"*100)
    else:
        print("Failed to fetch data.")

    #
    # if daily_model.fetch_data():
    #     # print("Data fetched successfully.")
    #     daily_model.analyze_chaos_dynamics()
    #     # ARIMA
    #     if daily_model.train_arima_model():
    #         # print("ARIMA Model trained successfully.")
    #         arima_forecast = daily_model.make_forecast(steps=forwardD, model_type='arima')
    #         print(f"\n{forwardD}-day ARIMA forecast: \n{LIGHT_CYAN}{arima_forecast}{END}")
    #         arima_rmse = daily_model.evaluate_model(model_type='arima')
    #         print(f"ARIMA Model RMSE: {LIGHT_CYAN}{arima_rmse:.8f}\n{END}")
    #         amean = arima_forecast.mean()
    #         print(f"ARIMA mean: {amean:.8f}")
    #         sub = amean-  arima_rmse
    #         add = arima_rmse + amean
    #         print(f"adder: {add:.8f} | subtract: {sub:.8f}")
        # SARIMA
        # if daily_model.train_sarima_model():
        #     # print("\nSARIMA Model trained successfully.")
        #     sarima_forecast = daily_model.make_forecast(steps=forwardD, model_type='sarima')
        #     print(f"{forwardD}-day SARIMA forecast: \n{LIGHT_CYAN}{sarima_forecast}{END}")
        #     sarima_rmse = daily_model.evaluate_model(model_type='sarima')
        #     print(f"SARIMA Model RMSE: {LIGHT_CYAN}{sarima_rmse:.8f}\n{END}")
        #     smean = sarima_forecast.mean()
        #     print(f"SARIMA mean: {smean:.8f}")
        # # GARCH
        # if daily_model.train_garch_model():
        #     # print("\nGARCH Model trained successfully.")
        #     garch_forecast = daily_model.make_forecast(steps=forwardD, model_type='garch')
        #     print(f"{forwardD}-day GARCH volatility forecast: \n{LIGHT_CYAN}{garch_forecast}{END}")
        #     garch_rmse = daily_model.evaluate_model(model_type='garch')
        #     print(f"GARCH Model RMSE: {LIGHT_CYAN}{garch_rmse:.8f}\n{END}")
        #     gmean = garch_forecast.mean()
        #     print(f"GARCH mean: {gmean:.8f}")
        # # VAR
        # if daily_model.train_var_model():
        #     # print("\nVAR Model trained successfully.")
        #     var_forecast = daily_model.make_forecast(steps=forwardD, model_type='var')
        #     print(f"{forwardD}-day VAR forecast: \n{LIGHT_CYAN}{var_forecast}{END}")
        #     var_rmse = daily_model.evaluate_model(model_type='var')
        #     print(f"VAR Model RMSE: {LIGHT_CYAN}{var_rmse:.8f}\n{END}")
        #     vmean = var_forecast.mean()
        #     print(f"VAR mean: {vmean:.8f}")
        #     print(f"-"*100)
    # else:
    #     print("Failed to fetch data.")

# # #----------------------------------------------------------------
#     if weekly_model.fetch_data():
#         # print("Data fetched successfully.")
#         weekly_model.analyze_chaos_dynamics()
#         # ARIMA
#         if weekly_model.train_arima_model():
#             # print("ARIMA Model trained successfully.")
#             arima_forecast = weekly_model.make_forecast(steps=forwardW, model_type='arima')
#             print(f"\n{forwardW}-week ARIMA forecast: \n{LIGHT_PURPLE}{arima_forecast}{END}")
#             arima_rmse = weekly_model.evaluate_model(model_type='arima')
#             print(f"ARIMA Model RMSE: {LIGHT_PURPLE}{arima_rmse:.8f}\n{END}")
#             amean = arima_forecast.mean()
#             print(f"ARIMA mean: {amean:.8f}")
#         # SARIMA
#         if weekly_model.train_sarima_model():
#             # print("\nSARIMA Model trained successfully.")
#             sarima_forecast = weekly_model.make_forecast(steps=forwardW, model_type='sarima')
#             print(f"{forwardW}-week SARIMA forecast: \n{LIGHT_PURPLE} {sarima_forecast}{END}")
#             sarima_rmse = weekly_model.evaluate_model(model_type='sarima')
#             print(f"SARIMA Model RMSE:{LIGHT_PURPLE} {sarima_rmse:.8f}\n{END}")
#             smean = sarima_forecast.mean()
#             print(f"SARIMA mean: {smean:.8f}")
#         # GARCH
#         if weekly_model.train_garch_model():
#             # print("\nGARCH Model trained successfully.")
#             garch_forecast = weekly_model.make_forecast(steps=forwardW, model_type='garch')
#             print(f"{forwardW}-week GARCH volatility forecast: \n{LIGHT_PURPLE} {garch_forecast}{END}")
#             garch_rmse = weekly_model.evaluate_model(model_type='garch')
#             print(f"GARCH Model RMSE:{LIGHT_PURPLE} {garch_rmse:.8f}\n{END}")
#             gmean = garch_forecast.mean()
#             print(f"GARCH mean: {gmean:.8f}")
#         # VAR
#         if weekly_model.train_var_model():
#             # print("\nVAR Model trained successfully.")
#             var_forecast = weekly_model.make_forecast(steps=forwardW, model_type='var')
#             print(f"{forwardW}-week VAR forecast: \n{LIGHT_PURPLE}{var_forecast}{END}")
#             var_rmse = weekly_model.evaluate_model(model_type='var')
#             print(f"VAR Model RMSE:{LIGHT_PURPLE} {var_rmse:.8f}\n{END}")
#             vmean = var_forecast.mean()
#             print(f"VAR mean: {vmean:.8f}")
#             print(f"-"*100)
#     else:
#         print("Failed to fetch data.")
#
# # #------------------------------------------------------------
#     if monthly_model.fetch_data():
#         # print("Data fetched successfully.")
#         monthly_model.analyze_chaos_dynamics()
#         # ARIMA
#         if monthly_model.train_arima_model():
#             # print("ARIMA Model trained successfully.")
#             arima_forecast = monthly_model.make_forecast(steps=forwardM, model_type='arima')
#             print(f"\n{forwardM}-month ARIMA forecast: \n{GREEN}{arima_forecast}{END}")
#             arima_rmse = monthly_model.evaluate_model(model_type='arima')
#             print(f"ARIMA Model RMSE: {GREEN}{arima_rmse:.8f}\n{END}")
#             amean = arima_forecast.mean()
#             print(f"ARIMA mean: {amean:.8f}")
#         # SARIMA
#         if monthly_model.train_sarima_model():
#             # print("\nSARIMA Model trained successfully.")
#             sarima_forecast = monthly_model.make_forecast(steps=forwardM, model_type='sarima')
#             print(f"{forwardM}-month SARIMA forecast: \n{GREEN}{sarima_forecast}{END}")
#             sarima_rmse = monthly_model.evaluate_model(model_type='sarima')
#             print(f"SARIMA Model RMSE: {GREEN}{sarima_rmse:.8f}\n{END}")
#             smean = sarima_forecast.mean()
#             print(f"SARIMA mean: {smean:.8f}")
#         # GARCH
#         if monthly_model.train_garch_model():
#             # print("\nGARCH Model trained successfully.")
#             garch_forecast = monthly_model.make_forecast(steps=forwardM, model_type='garch')
#             print(f"{forwardM}-month GARCH volatility forecast: \n{GREEN}{garch_forecast}{END}")
#             garch_rmse = monthly_model.evaluate_model(model_type='garch')
#             print(f"GARCH Model RMSE: {GREEN}{garch_rmse:.8f}\n{END}")
#             gmean = garch_forecast.mean()
#             print(f"GARCH mean: {gmean:.8f}")
#         # VAR
#         if monthly_model.train_var_model():
#             # print("\nVAR Model trained successfully.")
#             var_forecast = monthly_model.make_forecast(steps=forwardM, model_type='var')
#             print(f"{forwardM}-month VAR forecast: \n{LIGHT_GREEN}{var_forecast}{END}")
#             var_rmse = monthly_model.evaluate_model(model_type='var')
#             print(f"VAR Model RMSE: {GREEN}{var_rmse:.8f}\n{END}")
#             vmean = var_forecast.mean()
#             print(f"VAR mean: {vmean:.8f}")
#     else:
#         print("Failed to fetch data.")

# #------------------------------------------------------------


if __name__ == "__main__":
    while True:
        main()
        time.sleep(8)
