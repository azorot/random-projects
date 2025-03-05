from datafetchSrc import OHLCVDataFetcher
import time
from datetime import timedelta, datetime
import pandas as pd
import os
from colors import Colors as CL

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def hourly_job():
    # start_time = time.time()
    fetcher = OHLCVDataFetcher()
    symbol = 'LUNC/USDT'

    # Define target start date for fetching data
    target_start_date = datetime(2019, 7, 27, 0, 0, 0)
    target_start_timestamp = int(target_start_date.timestamp())

    # RESOLUTIONS
    resolutions = {
        '1': 'minute_df',
        # '5': 'minute5_df',
        # '15': 'minute15_df',
        # '30': 'minute30_df',
        # '60': 'hourly_df',
        # '240': 'hourly4_df',
        # '360': 'hourly6_df',
        # '720': 'hourly12_df',
        # '1D': 'daily_df',
        # '1W': 'weekly_df',
        # '1M': 'monthly_df'
    }

    for res, var_name in resolutions.items():
        csv_filename = f"{symbol.replace('/', '_')}_{var_name}.csv"

        # Check if the CSV file exists
        if os.path.exists(csv_filename):
            # Load existing data from CSV if it exists
            existing_df = pd.read_csv(csv_filename)
            print(f"{CL.CYAN}Loaded existing {var_name} from {csv_filename}{CL.END}")

            # Ensure timestamp is in datetime format
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])

            # Determine the last timestamp in the existing data
            last_existing_timestamp = existing_df['timestamp'].max()

            # Check if the existing data covers the required range
            if last_existing_timestamp >= target_start_date:
                print(f"{CL.GREEN}Existing data covers the required range. Fetching new data from {last_existing_timestamp}.{CL.END}")
                # Set from_timestamp as the last available timestamp + 1 second
                from_timestamp = int(last_existing_timestamp.timestamp()) + 1
            else:
                # If existing data doesn't cover the target range, fetch from target_start_date
                print(f"{CL.RED}Existing data doesn't cover the required range. Fetching from {target_start_date}.{CL.END}")
                from_timestamp = target_start_timestamp

        else:
            # If no file exists, set from_timestamp to target_start_timestamp
            print(f"{CL.RED}CSV file does not exist, starting from {target_start_date}.{CL.END}")
            from_timestamp = target_start_timestamp

        to_timestamp = int(time.time())  # Current time as end timestamp

        # Fetch new data from API
        new_data_df = fetcher.get_ohlcv_dataframe(symbol, res, from_timestamp, to_timestamp)

        if new_data_df is not None and not new_data_df.empty:
            # Ensure new data timestamps are in datetime format
            new_data_df['timestamp'] = pd.to_datetime(new_data_df['timestamp'])

            # Check if new data overlaps with existing data and remove duplicates
            if os.path.exists(csv_filename):
                # Combine the existing and new data, drop duplicates based on the timestamp
                combined_df = pd.concat([existing_df, new_data_df]).drop_duplicates(subset='timestamp', keep='last').sort_values('timestamp')
            else:
                combined_df = new_data_df

            # Save combined DataFrame back to CSV
            combined_df.to_csv(csv_filename, index=False)
            print(f"{CL.GREEN}Fetched and saved {var_name} to {csv_filename}{CL.END}")

        else:
            print(f"{CL.RED}No new data fetched for {var_name}{CL.END}")
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print(f"\nTotal execution time: {format_time(execution_time)}")

if __name__ == "__main__":
    hourly_job()  # Optional: Call this for testing or direct execution.
