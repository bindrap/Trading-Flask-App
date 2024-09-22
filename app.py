from flask import Flask, render_template, request
from pymongo import MongoClient
from alpha_vantage.timeseries import TimeSeries
import backtrader as bt
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# MongoDB Setup
client = MongoClient('mongodb://localhost:27017/')
db = client['stock_data']
collection = db['trades']

API_KEY = 'SLLD727OCBJQLYLM'

# Fetching dynamic top 5 gainers (e.g., using Yahoo Finance API, here we use static for example)
def fetch_top_gainers():
    # Dummy gainers for example, replace with screener API (e.g. Yahoo Finance, IEXCloud, etc.)
    top_gainers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    return top_gainers

# Fetch 1-minute data from Alpha Vantage
def fetch_alpha_vantage_data(symbol, interval='1min', outputsize='compact'):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, meta_data = ts.get_intraday(symbol=symbol, interval=interval, outputsize=outputsize)
    data.columns = ['open', 'high', 'low', 'close', 'volume']
    data = data.iloc[::-1]  # Reverse data order
    return data

# Bollinger Bands and MACD Strategy
class BollingerMACDStrategy(bt.Strategy):
    def __init__(self):
        self.sma = bt.indicators.SimpleMovingAverage(self.data.close, period=20)
        self.bollinger = bt.indicators.BollingerBands(self.data.close)
        self.macd = bt.indicators.MACD(self.data.close)
        self.trades = []

    def next(self):
        # Restrict to last 3 days for signals
        if self.data.datetime.date(0) >= (datetime.now().date() - timedelta(days=3)):
            if self.data.close[0] < self.bollinger.lines.bot and self.macd.macd > self.macd.signal:
                self.buy(size=10)
                self.trades.append({
                    'Type': 'BUY',
                    'Price': self.data.close[0],
                    'Size': 50,
                    'Date': self.data.datetime.datetime(0)
                })
            elif self.data.close[0] > self.bollinger.lines.top and self.macd.macd < self.macd.signal:
                self.sell(size=10)
                self.trades.append({
                    'Type': 'SELL',
                    'Price': self.data.close[0],
                    'Size': 50,
                    'Date': self.data.datetime.datetime(0)
                })

    def log_trades(self):
        return self.trades

# Run backtest on stock data
def run_backtest(data, symbol):
    cerebro = bt.Cerebro()
    cerebro.addstrategy(BollingerMACDStrategy)

    # Convert pandas DataFrame to Backtrader data feed
    bt_data = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(bt_data)

    # Set initial portfolio value and commission
    cerebro.broker.set_cash(1000)
    cerebro.broker.setcommission(commission=0.001)

    # Run backtest
    strategy_instance = cerebro.run()[0]
    trades = strategy_instance.log_trades()

    return trades, cerebro.broker.getvalue()

# Predict end-of-day price using linear regression
def predict_end_of_day_price(data):
    data['time_in_minutes'] = np.arange(len(data))  # Time in minutes from start of day

    # Model to predict price based on time
    X = data['time_in_minutes'].values.reshape(-1, 1)
    y = data['close'].values

    model = LinearRegression()
    model.fit(X, y)

    # Predict for the time corresponding to the market close (e.g. 390 minutes for US markets)
    end_of_day_time = 390
    predicted_price = model.predict([[end_of_day_time]])[0]

    return predicted_price

# Plot with buy/sell signals, Bollinger, MACD
def create_trade_plot(data, trades):
    fig = go.Figure()

    # Candlestick plot
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['open'],
                                 high=data['high'],
                                 low=data['low'],
                                 close=data['close'],
                                 name='Candlestick'))

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=data.index, y=data['close'].rolling(window=20).mean(),
                             mode='lines', line=dict(color='blue'), name='Bollinger SMA'))
    fig.add_trace(go.Scatter(x=data.index, y=data['close'].rolling(window=20).mean() + 2 * data['close'].rolling(window=20).std(),
                             mode='lines', line=dict(color='green'), name='Bollinger Upper'))
    fig.add_trace(go.Scatter(x=data.index, y=data['close'].rolling(window=20).mean() - 2 * data['close'].rolling(window=20).std(),
                             mode='lines', line=dict(color='red'), name='Bollinger Lower'))

    # MACD Plot (Example: Simple Moving Averages)
    fig.add_trace(go.Scatter(x=data.index, y=data['close'].ewm(span=12).mean() - data['close'].ewm(span=26).mean(),
                             mode='lines', line=dict(color='orange'), name='MACD'))

    # Buy/Sell signals
    for trade in trades:
        if trade['Type'] == 'BUY':
            fig.add_trace(go.Scatter(x=[trade['Date']], y=[trade['Price']],
                                     mode='markers', marker=dict(color='green', size=10),
                                     name='Buy Signal'))
        elif trade['Type'] == 'SELL':
            fig.add_trace(go.Scatter(x=[trade['Date']], y=[trade['Price']],
                                     mode='markers', marker=dict(color='red', size=10),
                                     name='Sell Signal'))

    fig.update_layout(title='Stock Price with Buy/Sell Signals', xaxis_title='Date', yaxis_title='Price')
    return fig.to_html()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/top-gainers')
def top_gainers():
    stock_symbols = fetch_top_gainers()
    trades_info = []

    for symbol in stock_symbols:
        data = fetch_alpha_vantage_data(symbol, '1min')
        trades, ending_value = run_backtest(data, symbol)
        predicted_price = predict_end_of_day_price(data)
        plot_html = create_trade_plot(data, trades)
        trades_info.append({
            'symbol': symbol,
            'plot': plot_html,
            'predicted_price': predicted_price
        })

    return render_template('gainers.html', trades_info=trades_info)

@app.route('/search', methods=['GET'])
def search_stock():
    symbol = request.args.get('symbol')
    data = fetch_alpha_vantage_data(symbol, '1min')
    trades, ending_value = run_backtest(data, symbol)
    predicted_price = predict_end_of_day_price(data)
    plot_html = create_trade_plot(data, trades)

    return render_template('stock.html', symbol=symbol, plot=plot_html, predicted_price=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)
