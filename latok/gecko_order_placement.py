import os
import time
import random
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import TimeoutException
import pandas as pd
from phemex_api_keys import phemex_email, phemex_pass
from colors import Colors as CL

# Constants
SYMBOL = 'LUNC/USDT'
CSV_FILENAME = f"{SYMBOL.replace('/', '_')}_minute_df.csv"
PER_ORDER = 500000  # Amount allowed for each order PnL (in USD)
last_mean = None
all_means = []

profile_path = '/home/azoroth/.mozilla/firefox/gt4dwpe1.default-release'
options = Options()
options.add_argument(f'-profile {profile_path}')
driver = webdriver.Firefox(options=options)

def random_sleep(min_seconds=2, max_seconds=5):
    '''Sleep for a random amount of time to simulate human behavior.'''
    sleep_time = random.uniform(min_seconds, max_seconds)
    print(f"Sleeping for {sleep_time:.2f} seconds")
    time.sleep(sleep_time)

def wait_for_element(xpath, timeout=30, clickable=False):
    '''Wait for an element to be visible (or clickable).'''
    try:
        if clickable:
            element = WebDriverWait(driver, timeout).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
        else:
            element = WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.XPATH, xpath))
            )
        return element
    except Exception as e:
        print(f"Error waiting for element {xpath}: {e}")
        return None

def read_csv():
    '''Read the minute_df CSV file and return a DataFrame.'''
    if os.path.exists(CSV_FILENAME):
        df = pd.read_csv(CSV_FILENAME)
        df['timestamp'] = pd.to_datetime(df['timestamp'])  # Ensure timestamp is in datetime format
        return df
    else:
        print(f"CSV file {CSV_FILENAME} does not exist.")
        return None

def calculate_mean_close(df):
    '''Calculate the mean of the 'close' column.'''
    if 'close' in df.columns:
        mean_close = df['close'].mean()
        return mean_close
    else:
        print("The 'close' column is missing from the DataFrame.")
        return None

def cancel_orders():
    '''Cancel all open orders.'''
    cancel_button = wait_for_element('//button[contains(text(), "Cancel All")]', timeout=30, clickable=True)
    if cancel_button:
        cancel_button.click()
        print("Clicked 'Cancel All' button.")
        # Wait for modal confirmation...

def switch_to_sell():
    '''Switch to sell position (if needed).'''
    sell_button = wait_for_element('//button[contains(@class, "sell-btn")]', timeout=30, clickable=True)
    if sell_button:
        sell_button.click()
        print("Switched to sell position.")
        random_sleep()

def place_sell_order(order_size):
    global last_mean  # Use the global variable to track the last mean

    if last_mean is not None:
        # Calculate the mean of the 'close' column
        mean_close = calculate_mean_close(read_csv())
        if mean_close is not None:
            # Compare current mean_close with last_mean
            epsilon = 1e-8  # Tolerance for floating-point comparison
            if abs(mean_close - last_mean) > epsilon:  # If the mean has changed
                print(f"Mean changed from {last_mean:.8f} to {mean_close:.8f} {CL.RED}change: {mean_close - last_mean:.8f}{CL.END}")
                return  # Skip order placement, exit function
            else:
                print(f"{CL.GREEN}Mean remains unchanged.{CL.END} {mean_close:.8f}")

                # Proceed with order placement since the mean has not changed
                rounded_price = round(mean_close, 8)  # Use rounded mean as price
                order_size = PER_ORDER / rounded_price
                rounded_size = int(round(order_size, 0)) - 1  # Calculate order size

                print(f"Placing sell order at price: {rounded_price}, Order size: {rounded_size} units")

                try:
                    price_input = wait_for_element('//div[contains(@class, "limit-price")]//input[@type="text"]', timeout=30)
                    qty_input = wait_for_element('//div[contains(@class, "limit-quantity")]//input[@type="text"]', timeout=30)

                    if price_input and qty_input:
                        price_input.clear()
                        price_input.send_keys(str(rounded_price))
                        qty_input.clear()
                        qty_input.send_keys(str(rounded_size))

                        sell_button = wait_for_element('//div[contains(@class, "btn-sell")]', timeout=30)
                        if sell_button:
                            sell_button.click()
                            print("Sell order placed.")
                    else:
                        print("Price or quantity input not found.")
                except Exception as e:
                    print(f"Error occurred during order placement: {str(e)}")

                # Update last_mean with the current mean
                last_mean = mean_close  # Set the last mean to the current mean

    else:
        print("No last mean available, skipping order placement.")


def fetch_webpage():
    driver.get('https://phemex.com/spot/trade/LUNCUSDT')
    print("Navigated to LUNC/USDT trading page.")
    random_sleep()

def main():
    global last_mean  # Declare last_mean as global

    df = read_csv()
    if df is not None:
        mean_close = calculate_mean_close(df)

        if mean_close is not None:
            all_means.append(mean_close)  # Store current mean in all_means

            rounded_price = round(mean_close, 8)  # Round the price to 8 decimals
            order_size = PER_ORDER / rounded_price

            rounded_size = int(round(order_size, 0)) - 1  # Calculate order size based on rounded price

            print(f"Order size per order: {rounded_size} units at price: {rounded_price}")

            place_sell_order(rounded_size)

            last_mean = mean_close  # Update last_mean with the current mean

if __name__ == "__main__":
    main()
