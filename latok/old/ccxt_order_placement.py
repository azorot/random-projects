import pandas as pd
import os
import ccxt
import time
from phemex_api_keys import phemex_id, phemex_secret

# Constants
SYMBOL = 'LUNC/USDT'
CSV_FILENAME = f"{SYMBOL.replace('/', '_')}_minute_df.csv"
PER_ORDER = 5000  # Amount for each order (in USD)

# Phemex API credentials
API_KEY = phemex_id
API_SECRET = phemex_secret

# Initialize CCXT Phemex client
exchange = ccxt.phemex({
    'apiKey': API_KEY,
    'secret': API_SECRET,
})

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

def get_available_balance():
    """Fetch the available LUNC balance."""
    try:
        balance = exchange.fetch_balance()
        lunc_balance = balance.get('total', {}).get('LUNC', 0)
        print(f"Available LUNC balance: {lunc_balance}")
        return lunc_balance
    except Exception as e:
        print(f"Error fetching balance: {e}")
        return 0

def get_existing_order():
    """Check if there are any existing orders."""
    try:
        # Fetch open orders from Phemex
        open_orders = exchange.fetch_open_orders(symbol=SYMBOL)
        if open_orders:
            print(f"Found existing order(s): {open_orders}")
            return open_orders
        else:
            print("No existing orders found.")
            return None
    except Exception as e:
        print(f"Error fetching open orders: {e}")
        return None

def cancel_order(order_id, mean_close):
    """Cancel an existing order by order_id."""
    try:
        rounded_price = round(mean_close, 8)  # Round the price to 8 decimals for cancellation
        exchange.cancel_order(order_id, SYMBOL)
        print(f"Canceled order {order_id}.")
    except Exception as e:
        print(f"Error canceling order: {e}")

def place_order(mean_close, order_size):
    """Place an order on Phemex with POST ONLY status."""
    try:
        # Round the price to 8 decimals, but don't round the order size
        rounded_price = round(mean_close, 8)  # Round price to 8 decimals
        print(f"Placing order at price: {rounded_price}, Order size: {order_size} units")

        # Place a limit order with "Post Only" status
        order = exchange.create_limit_buy_order(SYMBOL, order_size, rounded_price, {
            'postOnly': True  # Ensures it's a Post-Only order
        })
        print(f"Placed order: {order}")
    except Exception as e:
        print(f"Error placing order: {e}")

def amend_order(existing_order, mean_close, order_size):
    """Amend the existing order (cancel and place a new one)."""
    if existing_order:
        cancel_order(existing_order['id'], mean_close)  # Cancel the existing order
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
            existing_orders = get_existing_order()

            # Place orders iteratively until you run out of LUNC
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

                # Fetch the updated balance after placing the order
                available_lunc = get_available_balance()

                # Stop if there's not enough LUNC to place another order
                if available_lunc < order_size:
                    print(f"Not enough LUNC to place another order. Stopping.")
                    break

                # Wait between orders to avoid rate limits
                time.sleep(1)  # Sleep for 1 second before placing another order

if __name__ == "__main__":
    main()
