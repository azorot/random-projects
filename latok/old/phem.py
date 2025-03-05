import ccxt as c
import pandas as pd
from APIKEYS import API_ID, APISEC
import os
import time

# Your API credentials
api_key = API_ID
api_secret = APISEC

# Define parameters
symbol = 'LUNC/USDT'
tf = '1m'
csv_file = 'ohlcv_data.csv'

# Initialize Phemex client
phem = c.phemex({
    'apiKey': api_key,
    'secret': api_secret,
})

# Function to fetch and save OHLCV data
def fetch_and_save_ohlcv(since=None):
    # Fetch OHLCV data with a specified start time
    fetcher = phem.fetch_ohlcv(symbol, tf, since=since)

    # Debugging: Print fetched data
    print("Fetched OHLCV data:", fetcher)

    # Convert to DataFrame
    df = pd.DataFrame(fetcher, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Filter out invalid timestamps (if any)
    df = df[df['timestamp'].between(0, 253402300799999)]  # Valid range for pandas

    # Convert timestamp to datetime (specifying milliseconds)
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')  # Specify unit as milliseconds
    except Exception as e:
        print("Error converting timestamps:", e)
        return  # Exit if there's an error

    # Load existing data if CSV file exists
    if os.path.exists(csv_file):
        existing_df = pd.read_csv(csv_file)
        existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])

        # Concatenate new data with existing data and drop duplicates based on timestamp
        df = pd.concat([existing_df, df]).drop_duplicates(subset='timestamp').reset_index(drop=True)

    # Save to CSV file only if new data exists
    if not df.empty:
        df.to_csv(csv_file, index=False)

# Function to fetch all available OHLCV data over multiple requests
def fetch_all_ohlcv():
    since = None  # Start from the latest available data or set a specific timestamp in milliseconds.

    while True:
        fetch_and_save_ohlcv(since)

        # Check for new rows fetched
        new_data_count = len(pd.read_csv(csv_file)) - (len(pd.read_csv(csv_file)) if since is None else 0)

        if new_data_count == 0:
            print("No new data fetched.")
            break

        # Update since for the next request (get the last timestamp in milliseconds)
        last_timestamp_str = pd.read_csv(csv_file)['timestamp'].iloc[-1]  # Get last timestamp as string

        # Convert string to datetime and then to timestamp in milliseconds
        last_timestamp_dt = pd.to_datetime(last_timestamp_str)  # Convert string to datetime
        since = int(last_timestamp_dt.timestamp() * 1000) + 60000  # Add one minute to ensure we get new data

        time.sleep(1)  # Sleep for a second to avoid hitting rate limits

# Run the function to fetch all available OHLCV data
fetch_all_ohlcv()
