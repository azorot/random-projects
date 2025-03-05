#datafetchSrc.py
import requests
import json
import time
import pandas as pd
from datetime import datetime, timezone
from colors import Colors as CL
# current_from = 0
class OHLCVDataFetcher:
    def __init__(self, base_url="https://api.latoken.com/v2/tradingview/history"):
        self.base_url = base_url
        self.earliest_min_date = int(datetime(2019, 7, 27, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        self.earliest_5min_date = int(datetime(2019, 7, 27, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        self.earliest_15min_date = int(datetime(2019, 7, 27, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        self.earliest_30min_date = int(datetime(2019, 7, 27, 0, 0, 0, tzinfo=timezone.utc).timestamp())
        self.earliest_hourly_date = int(datetime(2019, 7, 27, 0, 0, 0,tzinfo=timezone.utc).timestamp())
        self.earliest_daily_date = int(datetime(2019, 7, 27).timestamp())

        # Define chunk sizes for different resolutions
        self.chunk_sizes = {
            "1": 29999,
            "5": 29999,
            "15": 29999,
            "30": 29999,
            "60": 29999,
            "240": 29999,
            "360": 29999,
            "720": 29999,
            "1D": 29999,
            "1W": 29999,
            "1M": 29999,
        }

    def fetch_ohlcv_data(self, symbol, resolution, from_timestamp, to_timestamp):
        params = {
            "symbol": symbol,
            "resolution": resolution,
            "from": from_timestamp,
            "to": to_timestamp
        }
        response = requests.get(self.base_url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 's' in data and data['s'] == 'error':
                print(f"{CL.RED}API Error: {data.get('errmsg', 'Unknown error')}{CL.END}")
                return None
            return data
        else:
            print(f"{CL.RED}HTTP Error: {response.status_code}, {response.text}{CL.END}")
            return None

    def process_ohlcv_data(self, data):
        if data and all(key in data for key in ['t', 'o', 'h', 'l', 'c', 'v']):
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(data['t'], unit='s'),
                'open': pd.to_numeric(data['o'], errors='coerce'),
                'high': pd.to_numeric(data['h'], errors='coerce'),
                'low': pd.to_numeric(data['l'], errors='coerce'),
                'close': pd.to_numeric(data['c'], errors='coerce'),
                'volume': pd.to_numeric(data['v'], errors='coerce')
            })
            return df
        else:
            print(f"{CL.RED}Error: Invalid data format{CL.END}")
            return None

    def get_ohlcv_dataframe(self, symbol, resolution, from_timestamp=None, to_timestamp=None):
        # Set default timestamps based on resolution
        if resolution == "1":
            if from_timestamp is None:
                from_timestamp = self.earliest_min_date
        elif resolution == "5":
            if from_timestamp is None:
                from_timestamp = self.earliest_5min_date
        elif resolution == "15":
            if from_timestamp is None:
                from_timestamp = self.earliest_15min_date
        elif resolution == "30":
            if from_timestamp is None:
                from_timestamp = self.earliest_30min_date
        elif resolution in ["60", "240", "360", "720"]:
            if from_timestamp is None:
                from_timestamp = self.earliest_hourly_date
        else:
            if from_timestamp is None:
                from_timestamp = self.earliest_daily_date

        # Ensure to_timestamp reflects current time or specified end date
        if to_timestamp is None:
            to_timestamp = int(time.time())

        all_data = []

        current_from = max(from_timestamp, self.earliest_min_date)

        while current_from < to_timestamp:
            chunk_size_minutes = self.chunk_sizes[resolution]
            current_to = min(current_from + chunk_size_minutes * (60 if resolution in ["1", "5", "15", "30"] else (24 * (60 if resolution == "60" else (30)))), to_timestamp)

            print(f"{CL.BLUE}Fetching {resolution} data from {datetime.fromtimestamp(current_from)} to {datetime.fromtimestamp(current_to)} from {self.base_url}{CL.END}")

            chunk_data = self.fetch_ohlcv_data(symbol, resolution, current_from, current_to)

            if chunk_data:
                chunk_df = self.process_ohlcv_data(chunk_data)
                if chunk_df is not None and not chunk_df.empty:
                    all_data.append(chunk_df)
                else:
                    print(f"{CL.RED}No data received for the period {datetime.fromtimestamp(current_from)} to {datetime.fromtimestamp(current_to)}{CL.END}")
            else:
                print(f"{CL.RED}Failed to fetch data for the period {datetime.fromtimestamp(current_from)} to {datetime.fromtimestamp(current_to)}{CL.END}")

            current_from = current_to + 1
            time.sleep(0.1) # Delay to avoid hitting rate limits

        if all_data:
            final_df = pd.concat(all_data, ignore_index=True).drop_duplicates().sort_values('timestamp')
            return final_df

        return None

    def get_latest_candle(self, df):
        if df is not None and not df.empty:
            latest_candle = df.iloc[-1].copy()
            latest_candle = self.process_latest_candle(latest_candle)
            return latest_candle
        return None

    def process_latest_candle(self, latest_candle):
        latest_candle = latest_candle.ffill()
        latest_candle['is_complete'] = False
        return latest_candle

#usage
#from datafetchSrc import OHLCVDataFetcher
# def main():
#     fetcher = OHLCVDataFetcher()
#     symbol = "LUNC/USDT"
#     daily = "1d"
#
#     historical_data, latest_candle = fetcher.get_ohlcv_dataframe(symbol, daily)
#
#     if historical_data is not None and latest_candle is not None:
#         latest_candle = fetcher.process_latest_candle(latest_candle)
#         print(f"Historical Data:\n{historical_data}")
#         print(f"Latest Candle:\n{latest_candle}")
#
# if __name__ == "__main__":
#     main()

