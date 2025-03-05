import time
from datetime import timedelta, datetime
import schedule
from csv_fetch import hourly_job  # Import the function directly
from gecko_order_placement import main as order_placement_main
from gecko_order_placement import fetch_webpage, switch_to_sell
# from phemex_order_placement import main as phemex_order_placement_main
def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))

def scheduled_job():
    start_time= time.time()
    """Function to run the CSV fetcher and then place orders."""
    print("Starting CSV fetcher...")
    hourly_job()  # Call the function directly to fetch data
    end_time_csv = time.time()
    execution_time_csv = end_time_csv - start_time
    print(f"\nTotal csv_fetch.py execution time: {format_time(execution_time_csv)}")
    print("Starting order placement...")
    order_placement_main()  # Call the main function of order_placement.py
    end_time_order = time.time()
    execution_time_order = end_time_csv - start_time
    print(f"\nTotal gecko_order_placement.py execution time: {format_time(execution_time_order)}")
    total_end = execution_time_csv + execution_time_order
    print(f'\ntotal time for both.py: {format_time(total_end)}')
def wait_until_next_minute():
    """Function to wait until the start of the next minute (exactly on the first second)."""
    # Get the current time and calculate how many seconds are left until the start of the next minute
    current_time = time.time()
    seconds_until_next_minute = 60 - (current_time % 60)
    time.sleep(seconds_until_next_minute)  # Sleep for the time left until the start of the next minute

def main():
    while True:
        wait_until_next_minute()  # Wait until the start of the next minute

        print("Starting scheduled job...")
        scheduled_job()  # Run the CSV fetcher and place orders

if __name__ == "__main__":
    fetch_webpage()
    switch_to_sell()
    main()
