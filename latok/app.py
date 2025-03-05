import dash
from dash import dcc, html, Input, Output
import pandas as pd
import plotly.graph_objects as go

app = dash.Dash(__name__)

try:
    df = pd.read_csv('LUNC_USDT_minute_df.csv', parse_dates=['timestamp'])  # Correct file name
except FileNotFoundError:
    print("Error: CSV file 'LUNC_USDT_minute_df.csv' not found.")
    exit()

# Convert OHLC and volume to numeric, handling errors
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df.dropna(inplace=True)  # Remove rows with NaN values

# *** KEY FIX: Ensure timestamp is datetime64[ns] ***
df['timestamp'] = pd.to_datetime(df['timestamp'])  # Force datetime conversion

close_mean = df['close'].mean()

app.layout = html.Div([
    html.H1("Candlestick Chart with Close Mean"),
    dcc.Graph(id="candlestick-chart"),
])

@app.callback(
    Output("candlestick-chart", "figure"),
    Input("candlestick-chart", "relayoutData")
)
def update_chart(relayout_data):

    fig = go.Figure(data=[go.Candlestick(x=df['timestamp'],
                                         open=df['open'],
                                         high=df['high'],
                                         low=df['low'],
                                         close=df['close'],
                                         name="Candlestick"),
                           go.Scatter(x=df['timestamp'],
                                      y=[close_mean] * len(df),
                                      mode='lines',
                                      name='Close Mean',
                                      line=dict(color='blue', width=2))
                          ])

    fig.update_layout(title="Candlestick Chart",
                      xaxis_title="Timestamp",
                      yaxis_title="Price",
                      xaxis_rangeslider_visible=False)

    if relayout_data:
        fig.update_layout(relayout_data)

    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
