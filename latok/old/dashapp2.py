from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
from forecast_model3 import ForecastModel
from datafetchSrc import OHLCVDataFetcher
import dash
import numpy as np
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
    model.train_arima_model()
    model.train_sarima_model()
    model.train_garch_model()
    model.train_var_model()

# Layout of the app
app.layout = html.Div([
    html.H1("LUNC/USDT Price Forecast"),
    dcc.Graph(id='daily-chart'),
    dcc.Graph(id='weekly-chart'),
    dcc.Graph(id='monthly-chart'),
    html.Div(id='metrics-table'),
    dcc.Interval(
        id='interval-component',
        interval=3600*1000,  # Update every hour
        n_intervals=0
    )
])

def calculate_metrics(models):
    metrics = []
    for res, model in models.items():
        hurst = model.calculate_hurst_exponent()
        fractal_dim = model.calculate_fractal_dimension()
        lyapunov = model.estimate_lyapunov_exponent()
        metrics.append([res, f"{hurst:.4f}", f"{fractal_dim:.4f}", f"{lyapunov:.4f}"])
    return metrics

def create_metrics_table(models):
    metrics = calculate_metrics(models)
    return html.Table([
        html.Thead(html.Tr([html.Th("Resolution"), html.Th("Hurst Exponent"), html.Th("Fractal Dimension"), html.Th("Lyapunov Exponent")])),
        html.Tbody([html.Tr([html.Td(m) for m in metric]) for metric in metrics])
    ])
@app.callback(
    [Output('daily-chart', 'figure'),
     Output('weekly-chart', 'figure'),
     Output('monthly-chart', 'figure'),
     Output('metrics-table', 'children')],
    [Input('interval-component', 'n_intervals')]
)
def update_charts(n):
    figures = []

    for res in resolutions:
        model = models[res]
        df = model.data

        # Check if data is available
        if df is None or df.empty:
            print(f"No data available for {res}.")
            continue

        # Create candlestick trace
        candlestick = go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='OHLC'
        )

        # Create forecast traces for each model
        forecast_traces = []
        for model_type in ['arima', 'sarima', 'garch', 'var']:
            print(f"Making forecast using model type: {model_type}")
            forecast = model.make_forecast(steps=5, model_type=model_type)

            # Print forecast output
            print(f"{model_type.upper()} forecast: {forecast}")

            if forecast is None or np.isnan(forecast).any():
                print(f"Forecasting failed for {model_type}.")
                continue

            # Handle GARCH forecasts differently since it returns variance
            if model_type == 'garch':
                forecast_dates = pd.date_range(start=df.index[-1], periods=6, freq=res)[1:]
                forecast_trace = go.Scatter(
                    x=forecast_dates,
                    y=np.sqrt(forecast),  # Convert variance to standard deviation for plotting
                    mode='lines+markers',
                    name='GARCH Forecast (Volatility)',
                    line=dict(color='orange', dash='dash')
                )
            else:
                forecast_dates = pd.date_range(start=df.index[-1], periods=6, freq=res)[1:]
                forecast_trace = go.Scatter(
                    x=forecast_dates,
                    y=forecast.flatten(),  # Ensure correct shape for plotting
                    mode='lines+markers',
                    name=f'{model_type.upper()} Forecast',
                    line=dict(dash='dash')
                )

            forecast_traces.append(forecast_trace)

        # Create the figure with all traces
        fig = go.Figure(data=[candlestick] + forecast_traces)
        fig.update_layout(
            title=f'{res} LUNC/USDT Price and Forecasts',
            xaxis_title='Date',
            yaxis_title='Price',
            yaxis_type="log",
            yaxis_tickformat=".8f",
            xaxis_rangeslider_visible=False
        )

        figures.append(fig)

    # Create metrics table
    metrics_table = create_metrics_table(models)

    return figures + [metrics_table]


if __name__ == '__main__':
    app.run_server(debug=True)
