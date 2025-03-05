from datafetchSrc import OHLCVDataFetcher
import time
from datetime import timedelta, datetime, timezone
import pandas as pd
import os
from colors import Colors as CL
import schedule

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))
def hourly_job():
    start_time = time.time()
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
    all_means = []
    myLunc = 2878380
    perOrder = 5000000
    # Fetch and save data for each resolution
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
                print(f"{CL.GREEN}Existing data covers the required range. No need to fetch new data.{CL.END}")
                # Set from_timestamp as the last available timestamp + 1 second
                from_timestamp = int(last_existing_timestamp.timestamp()) + 1
            else:
                # If existing data doesn't cover the target range, fetch from target_start_date
                print(f"{CL.RED}Existing data doesn't cover the required range. Fetching from {target_start_date}.{CL.END}")
                from_timestamp = target_start_timestamp

        else:
            # If no file exists, set from_timestamp to target_start_timestamp
            from_timestamp = target_start_timestamp

        to_timestamp = int(time.time())  # Current time as end timestamp

        # Fetch new data from API
        # print(f'{CL.YELLOW}Fetching {res} ohlcv data starting from {datetime.fromtimestamp(from_timestamp)}{CL.END}')
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

        # After CSV update, load the data into a DataFrame for analysis
        df = pd.read_csv(csv_filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])  # Ensure timestamp is in datetime format
        close_df = df['close']

        # Calculate and print the mean for this resolution
        resolution_mean = close_df.mean()
        all_means.append(resolution_mean)
        print(f'mean of resolution {res}: {CL.RED}{resolution_mean:.8f}{CL.END}')
        pnl = myLunc * resolution_mean
        order_amt = perOrder / resolution_mean
        print(f'potential PnL: {CL.GREEN}{pnl:,.2f}{CL.END}')
        print(f'per Order: {CL.YELLOW}{order_amt}{CL.END} ')

    # Calculate the overall mean of the means
    if all_means:
        overall_mean = sum(all_means) / len(all_means)
        print(f"\nOverall mean of all 'close' column means:\n{CL.RED}{overall_mean:.8f}{CL.END}")
        overall_pnl = myLunc * overall_mean
        overall_order_amt = perOrder / overall_mean
        print(f'potential PnL: {CL.GREEN}{overall_pnl:,.2f}{CL.END}')
        print(f'per Order: {CL.YELLOW}{overall_order_amt}{CL.END} ')
    else:
        print("No means to calculate the overall mean.")


    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nTotal execution time: {format_time(execution_time)}")

def wait_until_next_minute():
    """Function to wait until the start of the next minute (exactly on the first second)."""
    # Get the current time and calculate how many seconds are left until the start of the next minute
    current_time = time.time()
    seconds_until_next_minute = 60 - (current_time % 60)
    time.sleep(seconds_until_next_minute)
def main():
    print("Starting the hourly job scheduler...")
    # schedule.every(1).minute.do(hourly_job)

    while True:
        wait_until_next_minute()
        hourly_job()
        # schedule.run_pending()
        # time.sleep(1)

if __name__ == "__main__":
    main()
