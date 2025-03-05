#test.py
from datafetchSrc import OHLCVDataFetcher
import pandas as pd
import numpy as np
import time
from datetime import timedelta
import schedule
#------------------------------------------------------------------------------------------------
def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    # Calculate the Fast and Slow Exponential Moving Averages
    ema_fast = df['close'].ewm(span=fast_period, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow_period, adjust=False).mean()

    # Calculate the MACD line
    macd_line = ema_fast - ema_slow

    # Calculate the Signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

    # Calculate the MACD histogram
    macd_histogram = macd_line - signal_line

    # Add MACD indicators to the dataframe
    df['macd_line'] = macd_line
    df['signal_line'] = signal_line
    df['macd_histogram'] = macd_histogram

    return df

#------------------------------------------------------------------------------------------------

def hurst_exponent(time_series, max_lag=30):
    try:
        lags = range(2, min(max_lag, len(time_series) // 2))
        tau = [np.sqrt(np.std(np.subtract(time_series[lag:], time_series[:-lag]))) for lag in lags]

        if len(tau) < 2:  # Not enough data points
            return np.nan

        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0] * 2.0
    except Exception as e:
        print(f"Error in Hurst calculation: {e}")
        return np.nan

def add_hurst_exponent(df, column='close', window=100, max_lag=30):
    df['hurst_exponent'] = df[column].rolling(window=window).apply(
        lambda x: hurst_exponent(x.dropna().values, max_lag=max_lag), raw=False
    )
    return df

def lyapunov_exponent(time_series, max_lag=30):
    """Calculate the Lyapunov exponent for a given time series."""
    try:
        n = len(time_series)
        if n < 2:
            return np.nan

        # Prepare an array for distances
        distances = []

        for lag in range(1, min(max_lag, n // 2)):
            # Calculate distances between points
            for i in range(n - lag):
                d = abs(time_series[i + lag] - time_series[i])
                distances.append(d)

        # Calculate logarithm of distances
        log_distances = np.log(distances)

        if len(log_distances) < 2:
            return np.nan

        # Fit a line to log distances
        poly = np.polyfit(range(len(log_distances)), log_distances, 1)
        return poly[0]  # This is the Lyapunov exponent
    except Exception as e:
        print(f"Error in Lyapunov calculation: {e}")
        return np.nan
def add_lyapunov_exponent(df, column='close', window=100, max_lag=30):
    """Add a Lyapunov exponent feature to the DataFrame."""
    df['lyapunov_exponent'] = df[column].rolling(window=window).apply(
        lambda x: lyapunov_exponent(x.dropna().values, max_lag=max_lag), raw=False
    )
    return df
#--------------------------------------------------------------------
def calculate_window_and_lag(row_count):
    if row_count < 60:  # Very small dataset
        window_size = max(10, int(row_count * 0.1))  # At least 10% of rows
        max_lag = min(5, int(row_count * 0.2))       # At most 20% of rows
    elif row_count < 280:  # Small to medium dataset
        window_size = max(20, int(row_count * 0.1))  # At least 10% of rows
        max_lag = min(50, int(row_count * 0.15))     # At most 15% of rows
    elif row_count < 2000:  # Daily data (around 1986 rows)
        window_size = max(100, int(row_count * 0.05))  # At least 5% of rows
        max_lag = min(200, int(row_count * 0.1))       # At most 10% of rows
    else:  # Hourly data (around 24669 rows)
        window_size = max(250, int(row_count * 0.01))  # At least 1% of rows
        max_lag = min(500, int(row_count * 0.02))      # At most 2% of rows

    return window_size, max_lag

#-------------------
def interpret_hurst(h):
    if h < 0.45:
        return "Strong mean-reverting behavior"
    elif 0.45 <= h < 0.5:
        return "Weak mean-reverting behavior"
    elif 0.5 <= h < 0.55:
        return "Weak trending behavior"
    elif h >= 0.55:
        return "Strong trending behavior"
    else:
        return "Random walk (no clear trend)"

def interpret_lyapunov(l):
    if l < 0:
        return "Stable behavior (attracting fixed points)"
    elif l == 0:
        return "Neutral behavior (possibly periodic)"
    else:
        return "Chaotic behavior (sensitive dependence on initial conditions)"


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#--------------------------------------------------------------

def hourly_job():
    start_time = time.time()  # Start timing
    GREEN = "\033[92m"
    RESET = "\033[0m"
    fetcher = OHLCVDataFetcher()
    mylunc = 2878380.05
    orderamt = 5000000
    symbol = "LUNC/USDT"
    hourly ="60" #not 1h, not 1H
    daily = "1d"
    weekly= "1w"
    monthly = "1M"

    hourly_df = fetcher.get_ohlcv_dataframe(symbol, hourly)
    time.sleep(5)
    daily_df = fetcher.get_ohlcv_dataframe(symbol, daily)
    weekly_df = fetcher.get_ohlcv_dataframe(symbol, weekly)
    monthly_df = fetcher.get_ohlcv_dataframe(symbol, monthly)
    # #----------------------------------------------------------------
    # hourly_window_size, hourly_max_lag = calculate_window_and_lag(len(hourly_df))
    # daily_window_size, daily_max_lag = calculate_window_and_lag(len(daily_df))
    # weekly_window_size, weekly_max_lag = calculate_window_and_lag(len(weekly_df))
    # monthly_window_size, monthly_max_lag = calculate_window_and_lag(len(monthly_df))
    # #----------------------------------------------------------------
    # hourly_df = add_hurst_exponent(hourly_df, window=hourly_window_size, max_lag=hourly_max_lag)
    # daily_df = add_hurst_exponent(daily_df, window=daily_window_size, max_lag=daily_max_lag)
    # weekly_df = add_hurst_exponent(weekly_df, window=weekly_window_size, max_lag=weekly_max_lag)
    # monthly_df = add_hurst_exponent(monthly_df, window=monthly_window_size, max_lag=monthly_max_lag)
    # #----------------------------------------------------------------
    # hourly_df = add_lyapunov_exponent(hourly_df, window=hourly_window_size, max_lag=hourly_max_lag)
    # daily_df = add_lyapunov_exponent(daily_df, window=daily_window_size, max_lag=daily_max_lag)
    # weekly_df = add_lyapunov_exponent(weekly_df, window=weekly_window_size, max_lag=weekly_max_lag)
    # monthly_df = add_lyapunov_exponent(monthly_df, window=monthly_window_size, max_lag=monthly_max_lag)
#--------------------------------------------------------------------------------------------------------
    print(f"hourly_df:\n{hourly_df}")
    mean_hourly_open = hourly_df['open'].mean()
    mean_hourly_high = hourly_df['high'].mean()
    mean_hourly_low = hourly_df['low'].mean()
    mean_hourly_close = hourly_df['close'].mean()

    avg_hourly_mean = (mean_hourly_open + mean_hourly_high + mean_hourly_low + mean_hourly_close)/4
    profit_hourly = avg_hourly_mean * mylunc
    hourly_order_amt = orderamt/avg_hourly_mean
    #
    # hourly_hurst = hourly_df['hurst_exponent'].dropna().mean()
    # hourly_lyapunov = hourly_df['lyapunov_exponent'].dropna().mean()

    print(f"Mean of the 'open' column (hourly): {mean_hourly_open}")
    print(f"Mean of the 'high' column (hourly): {mean_hourly_high}")
    print(f"Mean of the 'low' column (hourly): {mean_hourly_low}")
    print(f"Mean of the 'close' column (hourly): {mean_hourly_close}")

    print(f"Average of the mean values (ohlc/4): {avg_hourly_mean:.8f}")
    print(f"Per order: {hourly_order_amt}")
    print(f"potential profit: {GREEN}{profit_hourly:,.2f}{RESET}")
    #
    # print(f"Daily Hurst Exponent (mean)(hourly): {hourly_hurst:.4f} - {interpret_hurst(hourly_hurst)}")
    # print(f"Daily Lyapunov Exponent (mean): {hourly_lyapunov:.4f} - {interpret_lyapunov(hourly_lyapunov)}")

#----------------------------------------------------------------
    print(f"daily_df:\n{daily_df}")
    mean_daily_open = daily_df['open'].mean()
    mean_daily_high = daily_df['high'].mean()
    mean_daily_low = daily_df['low'].mean()
    mean_daily_close = daily_df['close'].mean()

    avg_daily_mean = (mean_daily_open + mean_daily_high + mean_daily_low + mean_daily_close)/4
    profit_daily = avg_daily_mean * mylunc
    daily_order_amt = orderamt/avg_daily_mean

    # daily_hurst = daily_df['hurst_exponent'].dropna().mean()
    # daily_lyapunov = daily_df['lyapunov_exponent'].dropna().mean()

    print(f"Mean of the 'open' column (daily): {mean_daily_open}")
    print(f"Mean of the 'high' column (daily): {mean_daily_high}")
    print(f"Mean of the 'low' column (daily): {mean_daily_low}")
    print(f"Mean of the 'close' column (daily): {mean_daily_close}")

    print(f"Average of the mean values (ohlc/4): {avg_daily_mean:.8f}")
    print(f"Per order: {daily_order_amt}")
    print(f"potential profit: {GREEN}{profit_daily:,.2f}{RESET}")

    # print(f"Daily Hurst Exponent (mean)(daily): {daily_hurst:.4f} - {interpret_hurst(daily_hurst)}")
    # print(f"Daily Lyapunov Exponent (mean): {daily_lyapunov:.4f} - {interpret_lyapunov(daily_lyapunov)}")

#---------------------------------------------------------------
    print(f"weekly_df:\n{weekly_df}")
    mean_weekly_open = weekly_df['open'].mean()
    mean_weekly_high = weekly_df['high'].mean()
    mean_weekly_low = weekly_df['low'].mean()
    mean_weekly_close = weekly_df['close'].mean()

    avg_weekly_mean = (mean_weekly_open + mean_weekly_high + mean_weekly_low + mean_weekly_close)/4
    profit_weekly = avg_weekly_mean * mylunc
    weekly_order_amt = orderamt/avg_weekly_mean

    # weekly_hurst = weekly_df['hurst_exponent'].dropna().mean()
    # weekly_lyapunov = weekly_df['lyapunov_exponent'].dropna().mean()


    print(f"Mean of the 'open' column: {mean_weekly_open}")
    print(f"Mean of the 'high' column: {mean_weekly_high}")
    print(f"Mean of the 'low' column: {mean_weekly_low}")
    print(f"Mean of the 'close' column: {mean_weekly_close}")

    print(f"Average of the mean values (ohlc/4): {avg_weekly_mean:.8f}")
    print(f"Per order: {weekly_order_amt}")
    print(f"potential profit: {GREEN}{profit_weekly:,.2f}{RESET}")

    # print(f"Weekly Hurst Exponent (mean): {weekly_hurst:.4f} - {interpret_hurst(weekly_hurst)}")
    # print(f"weekly Lyapunov Exponent (mean): {weekly_lyapunov:.4f} - {interpret_lyapunov(weekly_lyapunov)}")
#-----------------------------------------------------------------
    print(f"monthly_df:\n{monthly_df}")
    mean_monthly_open = monthly_df['open'].mean()
    mean_monthly_high = monthly_df['high'].mean()
    mean_monthly_low = monthly_df['low'].mean()
    mean_monthly_close = monthly_df['close'].mean()

    avg_monthly_mean = (mean_monthly_open + mean_monthly_high + mean_monthly_low + mean_monthly_close)/4
    profit_monthly = avg_monthly_mean * mylunc
    monthly_order_amt = orderamt/avg_monthly_mean

    # monthly_hurst = monthly_df['hurst_exponent'].dropna().mean()
    # monthly_lyapunov = monthly_df['lyapunov_exponent'].dropna().mean()

    print(f"Mean of the 'open' column: {mean_monthly_open}")
    print(f"Mean of the 'high' column: {mean_monthly_high}")
    print(f"Mean of the 'low' column: {mean_monthly_low}")
    print(f"Mean of the 'close' column: {mean_monthly_close}")

    print(f"Average of the mean values (ohlc/4): {avg_monthly_mean:.8f}")
    print(f"Per order: {monthly_order_amt}")
    print(f"potential profit: {GREEN} {profit_monthly:,.2f} {RESET}")

    # print(f"Monthly Hurst Exponent (mean): {monthly_hurst:.4f} - {interpret_hurst(monthly_hurst)}")
    # print(f"monthly Lyapunov Exponent (mean): {monthly_lyapunov:.4f} - {interpret_lyapunov(monthly_lyapunov)}")
    print('\n')

    total_avg = (avg_hourly_mean+avg_daily_mean+avg_weekly_mean+avg_monthly_mean)/4
    print(f"total averages: {total_avg:.8f}")
    total_order_amt = orderamt/total_avg
    print(f"total orderamt: {total_order_amt}")
    profit_total = total_avg * mylunc
    print(f"profit total_avg: {GREEN}{profit_total:,.2f}{RESET}")
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {format_time(execution_time)}")
    # schedule.every().hour.at(":00").do(hourly_job)

def main():
    print("Starting the hourly job scheduler...")
    schedule.every().hour.at(":01").do(hourly_job)
    schedule.every().hour.at(":16").do(hourly_job)
    schedule.every().hour.at(":31").do(hourly_job)
    schedule.every().hour.at(":46").do(hourly_job)

    while True:
        schedule.run_pending()
        time.sleep(1)
    #
    # # If you want to keep the detailed timing for specific parts:
    # print(f"Data fetching time: {format_time(fetch_end - fetch_start)}")
    # print(f"Window and lag calculation time: {format_time(calc_end - calc_start)}")
    # print(f"Hurst exponent calculation time: {format_time(hurst_end - hurst_start)}")

#-------------------------------------------------------------
#|||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||
#--------------------------------------------------------------
    # macd_daily_df = calculate_macd(daily_df)
    # macd_weekly_df = calculate_macd(weekly_df)
    # macd_monthly_df = calculate_macd(monthly_df)
    # print(f"macd daily: \n {macd_daily_df}")
    # print(f"macd weekly: \n {macd_weekly_df}")
    # print(f"macd monthly: \n {macd_monthly_df}")
        # print(f"viewing results for {resolution}")
# #-------------------------------------------------------------
#         print("\nClose Price:")
#         curr_close = df["close"].iloc[-1]
#         prev_close = df["close"].iloc[-2]
#         print(f"Current Close:{curr_close:.8f}")
#         if curr_close > prev_close:
#             diff_close = curr_close - prev_close
#             print(f"Bullish: +{diff_close:.8f}")
#         elif curr_close < prev_close:
#             diff_close = prev_close - curr_close  # Change this line
#             print(f"Bearish: -{diff_close:.8f}")
#         else:
#             print("No change")
# #------------------------------------------------------------
#         print("\nMACD Histogram:")
#         curr_hist = df['macd_histogram'].iloc[-1]
#         prev_hist = df['macd_histogram'].iloc[-2]
#         print(f"Current Hist:{curr_hist:.8f}")
#         if curr_hist > prev_hist:
#             diff_hist_bull = curr_hist - prev_hist
#             print(f"Bullish: +{diff_hist_bull:.8f}")
#         if curr_hist < prev_hist:
#             diff_hist_bear = curr_hist - prev_hist
#             print(f"Bearish: -{diff_hist_bear:.8f}")
# #------------------------------------------------------------
#         print("\nMACD Line:")
#         curr_macd = df['macd_line'].iloc[-1]
#         prev_macd = df['macd_line'].iloc[-2]
#         print(f"Current MACD:{curr_macd:.8f}")
#         if curr_macd > prev_macd:
#             diff_macd_bull = curr_macd - prev_macd
#             print(f"Bullish: +{diff_macd_bull:.8f}")
#         if curr_macd < prev_macd:
#             diff_macd_bear = curr_macd - prev_macd
#             print(f"Bearish: -{diff_macd_bear:.8f}")
# #-------------------------------------------------------------
#         print("\nSignal Line:")
#         curr_sig = df['signal_line'].iloc[-1]
#         prev_sig = df['signal_line'].iloc[-2]
#         print(f"Current Signal:{curr_sig:.8f}")
#         if curr_sig > prev_sig:
#             diff_sig_bull = curr_sig - prev_sig
#             print(f"Bullish: +{diff_sig_bull:.8f}\n")
#         if curr_sig < prev_sig:
#             diff_sig_bear = curr_sig - prev_sig
#             print(f"Bearish: -{diff_sig_bear:.8f}\n")
#
# #------------------------------------------------------------------
#         # Calculated currPrice
#         if curr_sig > 0:
#             calc_price = (curr_sig - curr_close)
#             print(f"Calculated sigPriceCurr: {calc_price:.8f}")
#         elif curr_sig < 0:
#             calc_price = (curr_sig + curr_close)
#             print(f"Calculated sigPriceCurr: {calc_price:.8f}")
#         else:
#             print("Signal is zero")
#
#         # Calculated prevPrice
#         if prev_sig > 0:
#             calc_price2 = (prev_sig - prev_close)
#             print(f"Calculated sigPricePrev: {calc_price2:.8f}")
#         elif prev_sig < 0:
#             calc_price2 = (prev_sig + prev_close)
#             print(f"Calculated sigPricePrev: {calc_price2:.8f}")
#         else:
#             print("Signal is zero")
#
#         #difference of currsigp and prevsigp
#         if calc_price > calc_price2:
#             diff_calcp = calc_price - calc_price2
#             print(f"Bullish: +{diff_calcp:.8f}")
#         elif calc_price < calc_price2:
#             diff_calcp = calc_price - calc_price2  # Change this line
#             print(f"Bearish: -{diff_calcp:.8f}")

if __name__ == "__main__":
    main()
