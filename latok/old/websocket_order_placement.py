import asyncio
import websockets
import json
import pandas as pd
import os
import time
import hmac
import hashlib
import requests
from phemex_api_keys import phemex_id, phemex_secret

# Constants
SYMBOL = 'LUNC/USDT'
CSV_FILENAME = f"{SYMBOL.replace('/', '_')}_minute_df.csv"
PER_ORDER = 5000000  # Amount for each order (in USD)

# Phemex API credentials
API_KEY = phemex_id
API_SECRET = phemex_secret

# Base URL for Phemex API
BASE_URL = 'https://api.phemex.com'

# WebSocket URL for Phemex
WS_URL = 'wss://phemex.com/ws'

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

def place_order(price, order_size):
    """Place an order on Phemex with POST ONLY status."""
    rounded_price = round(price, 8)  # Round price to 8 decimals
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

async def handle_ws_messages():
    """Handle WebSocket messages to place orders based on price conditions."""
    async with websockets.connect(WS_URL) as websocket:
        # Subscribe to market data for LUNC/USDT
        subscribe_msg = {
            "method": "orderbook.subscribe",
            "params": {
                "symbol": SYMBOL
            }
        }

        await websocket.send(json.dumps(subscribe_msg))

        # Fetch the available balance before starting the order process
        available_lunc = get_available_balance()

        while True:
            # Read the latest CSV data and calculate the mean close price
            df = read_csv()
            if df is not None:
                mean_close = calculate_mean_close(df)

                if mean_close is None:
                    print("Failed to calculate mean close, skipping this cycle.")
                    continue  # Skip this loop if the mean close is not available

            # Receive WebSocket message (price updates)
            message = await websocket.recv()
            data = json.loads(message)

            # Log received data
            print(f"Received data: {data}")

            # Check if we have the relevant price data
            if 'data' in data and 'ask' in data['data']:
                # Get the current ask price (best buy price)
                ask_price = float(data['data']['ask'][0][0])  # The first ask price
                print(f"Current Ask Price: {ask_price}")

                # Example: If price is below 100,000, place an order
                if ask_price < mean_close and available_lunc > 0:
                    # Calculate the order size based on PER_ORDER value in USD
                    order_size = PER_ORDER / ask_price
                    print(f"Order size per order: {order_size} units at price: {ask_price}")

                    # Place the order
                    place_order(ask_price, order_size)

                    # Reduce the available LUNC by the order size placed
                    available_lunc -= order_size
                    print(f"Remaining LUNC: {available_lunc}")

                    # Fetch updated balance after placing the order
                    time.sleep(1)
                    available_lunc = get_available_balance()

                    # Stop if there's not enough LUNC to place another order
                    if available_lunc < order_size:
                        print("Not enough LUNC to place another order. Stopping.")
                        break

            # Add some delay to avoid spamming WebSocket messages and rate limits
            await asyncio.sleep(1)

# Run the WebSocket handler
if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(handle_ws_messages())
