import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from forecast_model import ForecastModel
from datafetchSrc import OHLCVDataFetcher
import pandas as pd
# Initialize the Dash app
app = dash.Dash(__name__)

# Initialize the data fetcher and models
fetcher = OHLCVDataFetcher()
symbol = "LUNC/USDT"
resolutions = ["1d", "1w", "1M"]
models = {res: ForecastModel(symbol, res) for res in resolutions}

# Fetch data and train models
for res, model in models.items():
    model.fetch_data()
    model.train_model()

# Layout of the app
app.layout = html.Div([
    html.H1("LUNC/USDT Price Forecast"),
    dcc.Graph(id='daily-chart'),
    dcc.Graph(id='weekly-chart'),
    dcc.Graph(id='monthly-chart'),
    dcc.Interval(
        id='interval-component',
        interval=30*1000,  # Update every hour
        n_intervals=0
    )
])

# Callback to update charts
@app.callback(
    [Output('daily-chart', 'figure'),
     Output('weekly-chart', 'figure'),
     Output('monthly-chart', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_charts(n):
    figures = []
    for res in resolutions:
        model = models[res]
        df = model.data
        forecast = model.make_forecast(steps=60)

        # Create candlestick trace
        candlestick = go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        )

        # Create forecast trace
        forecast_dates = pd.date_range(start=df.index[-1], periods=16, freq=res)[1:]
        forecast_trace = go.Scatter(
            x=forecast_dates,
            y=forecast,
            mode='lines+markers',
            name='Forecast',
            line=dict(color='blue', dash='dash')
        )

        # Create the figure
        fig = go.Figure(data=[candlestick, forecast_trace])
        fig.update_layout(
            title=f'{res} LUNC/USDT Price and Forecast',
            xaxis_title='Date',
            yaxis_title='Price',
            yaxis_type="log",
            yaxis_tickformat=".8f",
            xaxis_rangeslider_visible=False
        )
        figures.append(fig)

    return figures

if __name__ == '__main__':
    app.run_server(debug=True)
