import pandas as pd
import os
import time
import hmac
import hashlib
import json
import requests
from phemex_api_keys import phemex_id, phemex_secret

# Constants
SYMBOL = 'LUNC/USDT'
CSV_FILENAME = f"{SYMBOL.replace('/', '_')}_minute_df.csv"
PER_ORDER = 500000  # Amount for each order (in USD)

# Phemex API credentials
API_KEY = phemex_id
API_SECRET = phemex_secret

# Base URL for Phemex API
BASE_URL = 'https://api.phemex.com'

def generate_signature(api_secret, params):
    """Generate a signature for the API request."""
    query_string = '&'.join([f"{key}={value}" for key, value in sorted(params.items())])
    return hmac.new(api_secret.encode(), query_string.encode(), hashlib.sha256).hexdigest()

def read_csv():
    """Read the minute_df CSV file and return a DataFrame."""
    if os.path.exists(CSV_FILENAME):
        df = pd.read_csv(CSV_FILENAME)
        df['timestamp'] = pd.to_datetime(df['timestamp'])  # Ensure timestamp is in datetime format
        return df
    else:
        print(f"CSV file {CSV_FILENAME} does not exist.")
        return None

def calculate_mean_close(df):
    """Calculate the mean of the 'close' column."""
    if 'close' in df.columns:
        mean_close = df['close'].mean()
        return mean_close
    else:
        print("The 'close' column is missing from the DataFrame.")
        return None

def api_request(method, endpoint, params=None):
    """Generic function to handle API requests with rate limit handling."""
    retries = 3
    for attempt in range(retries):
        try:
            if method == 'GET':
                response = requests.get(BASE_URL + endpoint, headers={'X-Phemex-API-Key': API_KEY}, params=params)
            elif method == 'POST':
                response = requests.post(BASE_URL + endpoint, headers={'X-Phemex-API-Key': API_KEY}, json=params)

            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limit exceeded
                retry_after = int(response.headers.get('x-ratelimit-retry-after', 60))  # Default to 60 seconds if not specified
                print(f"Rate limit exceeded. Retrying after {retry_after} seconds...")
                time.sleep(retry_after)
            else:
                print(f"Error fetching data: {response.json()}")
                return None

        except Exception as e:
            print(f"Error during API request: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff

def get_available_balance():
    """Fetch the available LUNC balance."""
    endpoint = '/v1/account/assets'
    params = {
        'symbol': 'LUNC',
        'timestamp': int(time.time() * 1000)
    }
    params['sign'] = generate_signature(API_SECRET, params)

    balance_data = api_request('GET', endpoint, params)

    if balance_data:
        lunc_balance = balance_data.get('data', {}).get('total', {}).get('LUNC', 0)
        print(f"Available LUNC balance: {lunc_balance}")
        return lunc_balance

    return 0

def get_existing_orders():
    """Check if there are any existing orders."""
    endpoint = '/v1/order/open'
    params = {
        'symbol': SYMBOL,
        'timestamp': int(time.time() * 1000)
    }
    params['sign'] = generate_signature(API_SECRET, params)

    open_orders_data = api_request('GET', endpoint, params)

    if open_orders_data:
        open_orders = open_orders_data.get('data', [])
        if open_orders:
            print(f"Found existing order(s): {open_orders}")
            return open_orders
        else:
            print("No existing orders found.")
            return None

    return None

def cancel_order(order_id):
    """Cancel an existing order by order_id."""
    endpoint = '/v1/order/cancel'
    params = {
        'orderID': order_id,
        'timestamp': int(time.time() * 1000)
    }
    params['sign'] = generate_signature(API_SECRET, params)

    response_data = api_request('POST', endpoint, params)

    if response_data and response_data.get('code') == 200:
        print(f"Canceled order {order_id}.")
    else:
        print(f"Error canceling order: {response_data}")

def place_order(mean_close, order_size):
    """Place an order on Phemex with POST ONLY status."""
    rounded_price = round(mean_close, 8)  # Round price to 8 decimals
    print(f"Placing order at price: {rounded_price}, Order size: {order_size} units")

    # Place a limit buy order
    endpoint = '/v1/order/create'
    params = {
        'symbol': SYMBOL,
        'price': rounded_price,
        'qty': order_size,
        'side': 'Buy',
        'ordType': 'Limit',
        'timeInForce': 'GoodTillCancel',
        'timestamp': int(time.time() * 1000)
    }

    params['sign'] = generate_signature(API_SECRET, params)

    response_data = api_request('POST', endpoint, params)

    if response_data and response_data.get('code') == 200:
        print(f"Placed order: {response_data}")
    else:
        print(f"Error placing order: {response_data}")

def amend_order(existing_order, mean_close, order_size):
    """Amend the existing order (cancel and place a new one)."""
    if existing_order:
        cancel_order(existing_order['orderID'])  # Cancel the existing order
    place_order(mean_close, order_size)  # Place a new order with the updated price and size

def main():
    """Main function to perform order placement."""
    df = read_csv()

    if df is not None:
        mean_close = calculate_mean_close(df)

        if mean_close is not None:
            # Fetch available LUNC balance
            available_lunc = get_available_balance()

            # Calculate the order size in units (based on PER_ORDER value in USD)
            rounded_price = round(mean_close, 8)  # Round the price to 8 decimals
            order_size = PER_ORDER / rounded_price  # Calculate the order size based on the rounded price
            print(f"Order size per order: {order_size} units at price: {rounded_price}")

            # Check for existing orders
            existing_orders = get_existing_orders()

            # Place orders iteratively until you run out of LUNC or reach rate limits
            while available_lunc > 0:
                # Ensure the order size does not exceed available LUNC balance
                if order_size > available_lunc:
                    order_size = available_lunc  # Adjust the order size to the remaining balance

                if existing_orders:
                    # If there's an existing order, amend it (cancel and place a new one)
                    for existing_order in existing_orders:
                        amend_order(existing_order, mean_close, order_size)
                else:
                    # No existing orders, so place a new one
                    place_order(mean_close, order_size)

                # Reduce the available LUNC by the order size placed
                available_lunc -= order_size
                print(f"Remaining LUNC: {available_lunc}")

                # Fetch updated balance after placing the order with delay to avoid hitting rate limits.
                time.sleep(1)
                available_lunc = get_available_balance()

                # Stop if there's not enough LUNC to place another order
                if available_lunc < order_size:
                    print("Not enough LUNC to place another order. Stopping.")
                    break

if __name__ == "__main__":
    main()
