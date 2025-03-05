from datafetchSrc import OHLCVDataFetcher
import time
from datetime import timedelta, datetime
import pandas as pd
import os
from colors import Colors as CL

# Global variable to store the last mean
last_mean = None

def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def hourly_job():
    global last_mean  # Use the global variable to track the last mean
    start_time = time.time()
    fetcher = OHLCVDataFetcher()
    symbol = 'LUNC/USDT'

    target_start_date = datetime(2019, 7, 27, 0, 0, 0)
    target_start_timestamp = int(target_start_date.timestamp())

    resolutions = {
        '1': 'minute_df',
    }

    all_means = []
    myLunc = 2926391.97
    perOrder = 5000000

    for res, var_name in resolutions.items():
        csv_filename = f"{symbol.replace('/', '_')}_{var_name}.csv"

        if os.path.exists(csv_filename):
            existing_df = pd.read_csv(csv_filename)
            existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
            last_existing_timestamp = existing_df['timestamp'].max()

            if last_existing_timestamp >= target_start_date:
                from_timestamp = int(last_existing_timestamp.timestamp()) + 1
            else:
                from_timestamp = target_start_timestamp
        else:
            from_timestamp = target_start_timestamp

        to_timestamp = int(time.time())
        new_data_df = fetcher.get_ohlcv_dataframe(symbol, res, from_timestamp, to_timestamp)

        if new_data_df is not None and not new_data_df.empty:
            new_data_df['timestamp'] = pd.to_datetime(new_data_df['timestamp'])
            if os.path.exists(csv_filename):
                combined_df = pd.concat([existing_df, new_data_df]).drop_duplicates(subset='timestamp', keep='last').sort_values('timestamp')
            else:
                combined_df = new_data_df

            combined_df.to_csv(csv_filename, index=False)

        df = pd.read_csv(csv_filename)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        close_df = df['close']

        resolution_mean = close_df.mean()
        all_means.append(resolution_mean)

        print(f'mean of resolution {res}: {CL.RED}{resolution_mean:.8f}{CL.END}')

        # Calculate potential PnL and order amount
        pnl = myLunc * resolution_mean
        order_amt = perOrder / resolution_mean

        print(f'potential PnL: {CL.GREEN}{pnl:,.2f}{CL.END}')
        print(f'per Order: {CL.YELLOW}{order_amt}{CL.END} ')

    if all_means:
        overall_mean = sum(all_means) / len(all_means)

        # Check for change in mean
        if last_mean is not None:
            if overall_mean != last_mean:
                change = overall_mean - last_mean
                print(f"Mean changed from {last_mean:.8f} to {overall_mean:.8f} {CL.RED}change:{change:.8f}{CL.END}")

            else:
                print(f"{CL.GREEN}Mean remains unchanged.{CL.END} {overall_mean}")

        last_mean = overall_mean  # Update last mean


    end_time = time.time()
    execution_time = end_time - start_time

    print(f"\nTotal execution time: {format_time(execution_time)}")

def wait_until_next_minute():
    current_time = time.time()
    seconds_until_next_minute = 60 - (current_time % 60)
    time.sleep(seconds_until_next_minute)

def main():
    print("Starting the hourly job scheduler...")

    while True:
        wait_until_next_minute()
        hourly_job()

if __name__ == "__main__":
    main()
