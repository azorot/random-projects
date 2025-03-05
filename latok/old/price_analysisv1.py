#price_analysisv1.py
from datafetchSrc import OHLCVDataFetcher
import time
from datetime import timedelta


def format_time(seconds):
    return str(timedelta(seconds=int(seconds)))


start_time= time.time()
fetcher = OHLCVDataFetcher()
symbol = 'LUNC/USDT'


# "supported_resolutions":["1","5","15","30","60","240","360","720","1D","1W","1M"]
#RESOLUTIONS
minute = '1'
minute5 = '5'
minute15 = '15'
minute30 = '30'

#HOURLY TIME FRAMES
hourly = '60'
hourly4 = '240'
hourly6 = '360'
hourly12 = '720'

#LARGER TIMEFRAMES
daily = '1d'
weekly= '1w'
monthly = '1M'


#MINUTE
print(f'Fetching 1min ohlcv data!')
minute_df = fetcher.get_ohlcv_dataframe(symbol, minute)
print(f'\n')
# print(f'Fetching 5min ohlcv data!')
# minute5_df =fetcher.get_ohlcv_dataframe(symbol, minute5)
# print(f'\n')
# print(f'Fetching 15min ohlcv data!')
# minute15_df=fetcher.get_ohlcv_dataframe(symbol, minute15)
# print(f'\n')
# print(f'Fetching 30min ohlcv data!')
# minute30_df=fetcher.get_ohlcv_dataframe(symbol, minute30)
# print(f'\n')
# #HOURLY
# print(f'Fetching 1hr ohlcv data!')
# hourly_df = fetcher.get_ohlcv_dataframe(symbol, hourly)
# print(f'\n')
# print(f'Fetching 4hr ohlcv data!')
# hourly4_df = fetcher.get_ohlcv_dataframe(symbol, hourly4)
# print(f'\n')
# print(f'Fetching 6hr ohlcv data!')
# hourly6_df = fetcher.get_ohlcv_dataframe(symbol, hourly6)
# print(f'\n')
# print(f'Fetching 12hr ohlcv data!')
# hourly12_df = fetcher.get_ohlcv_dataframe(symbol, hourly12)
#
# #LARGER TIMEFRAMES
# print(f'\n')
# print(f'Fetching Daily ohlcv data!')
# daily_df = fetcher.get_ohlcv_dataframe(symbol, daily)
# print(f'\n')
# print(f'Fetching Weekly ohlcv data!')
# weekly_df = fetcher.get_ohlcv_dataframe(symbol, weekly)
# print(f'\n')
# print(f'Fetching Monthly ohlcv data!')
# monthly_df = fetcher.get_ohlcv_dataframe(symbol, monthly)


print(f"minute_df: \n {minute_df}")
# print(f"minute5_df \n{minute5_df}")
# print(f"minute15_df \n{minute15_df}")
# print(f"minute30_df \n{minute30_df}")
#
# print(f"hourly_df: \n {hourly_df}")
# print(f"hourly4_df: \n {hourly4_df}")
# print(f"hourly6_df: \n {hourly6_df}")
# print(f"hourly12_df: \n {hourly12_df}")
#
# print(f'daily_df:\n{daily_df}')
# print(f'weekly_df:\n{weekly_df}')
# print(f'monthly_df:\n{monthly_df}')

end_time = time.time()

execution_time = end_time - start_time

print(f"\nTotal execution time: {format_time(execution_time)}")
