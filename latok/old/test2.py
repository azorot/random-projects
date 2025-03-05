#test2.py
from datafetchSrc import OHLCVDataFetcher
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# Set pandas option to avoid downcasting warning
pd.set_option('future.no_silent_downcasting', True)

def prepare_features(df):
    df = df.copy()
    df.loc[:, 'returns'] = df['close'].pct_change()
    df.loc[:, 'log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df.loc[:, 'volatility'] = df['returns'].rolling(window=20).std()
    return df.dropna()

def train_model(historical_data):
    data = prepare_features(historical_data)

    features = ['open', 'high', 'low', 'close', 'volume', 'returns', 'log_returns', 'volatility']
    X = data[features]
    y = data['close'].shift(-1)  # Predict next day's close

    X = X[:-1]
    y = y[:-1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    return model, scaler, features

def predict_next_close(model, scaler, features, historical_data, latest_candle):
    latest_data = pd.concat([historical_data.tail(20), pd.DataFrame([latest_candle])])
    latest_data = prepare_features(latest_data)
    latest_features = latest_data.iloc[-1][features]
    latest_features_scaled = pd.DataFrame(scaler.transform(latest_features.values.reshape(1, -1)), columns=features)
    prediction = model.predict(latest_features_scaled)
    return prediction[0]

def main():
    fetcher = OHLCVDataFetcher()
    symbol = "LUNC/USDT"
    daily = "1d"
    weekly = "1w"
    monthly = "1M"

    daily_df = fetcher.get_ohlcv_dataframe(symbol, daily)
    weekly_df = fetcher.get_ohlcv_dataframe(symbol, weekly)
    monthly_df = fetcher.get_ohlcv_dataframe(symbol, monthly)

    predictions = []

    for df, timeframe in [(daily_df, "daily"), (weekly_df, "weekly"), (monthly_df, "monthly")]:
        if df is not None:
            historical_data = df.iloc[:-1]  # All data except the last row
            latest_candle = fetcher.get_latest_candle(df)

            print(f"{timeframe.capitalize()} Historical Data:")
            print(historical_data.tail())
            print(f"\n{timeframe.capitalize()} Latest Candle:")
            print(latest_candle)

            # Train model on historical data
            model, scaler, features = train_model(historical_data)

            # Make prediction using the latest candle
            next_close_prediction = predict_next_close(model, scaler, features, historical_data, latest_candle)

            print(f"\nPredicted next close ({timeframe}): {next_close_prediction:.8f}")
            predictions.append(next_close_prediction)

    if predictions:
        average_prediction = sum(predictions) / len(predictions)
        print(f"Average prediction: {average_prediction:.8f}")
    else:
        print("No predictions were made due to lack of data.")

if __name__ == "__main__":
    main()
