import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

# Set the Matplotlib backend to 'Agg'
import matplotlib
matplotlib.use('Agg')

from flask import Flask, render_template
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf
import base64
from io import BytesIO
import matplotlib.dates as mdates
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.api import SimpleExpSmoothing

app = Flask(__name__)

@app.route('/')
def index():
    # Initialize the plots dictionary early
    plots = {}
    stock_data = pd.DataFrame()  # Initialize stock_data to avoid potential reference issues
    loss_plot = None  # Initialize loss_plot to avoid potential reference issues
    rmse, mape = None, None  # Initialize metrics

    # Load the CSV file
    try:
        stock_data = pd.read_csv('NFLX.csv', parse_dates=['Date'], index_col='Date')
        stock_data.dropna(inplace=True)

        # Show the first 10 rows of data
        first_data = stock_data.head(10)

        # Define input features (X) and target variable (y)
        x = stock_data[['Open', 'High', 'Low']]
        y = stock_data['Close']

        # Split the data into training and test sets
        x_ft, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=101)

        # Define the LSTM splitting function
        def lstm_split(data, time_steps):
            x, y = [], []
            for i in range(len(data) - time_steps):
                x.append(data[i:i + time_steps, :-1])
                y.append(data[i:i + time_steps, -1])
            return np.array(x), np.array(y)

        # Split the data for LSTM
        x1, y1 = lstm_split(x_ft.values, 60)

        train_split = 0.8
        split_idx = int(np.ceil(len(x1) * train_split))

        # Split into training and testing sets
        x_train, x_test = x1[:split_idx], x1[split_idx:]
        y_train, y_test = y1[:split_idx], y1[split_idx:]

        # Define and compile the LSTM model
        lstm = Sequential()
        lstm.add(LSTM(50, return_sequences=False, input_shape=(x_train.shape[1], x_train.shape[2])))
        lstm.add(Dense(60))
        lstm.compile(optimizer='adam', loss='mean_squared_error')

        # Fit the LSTM model
        history = lstm.fit(x_train, y_train,
                           epochs=100,
                           batch_size=32,
                           validation_data=(x_test, y_test),
                           verbose=2,
                           shuffle=False)

        # Make predictions
        y_pred = lstm.predict(x_test)

        # Calculate RMSE and MAPE
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = mean_absolute_percentage_error(y_test, y_pred)

        print(f'RMSE: {rmse}')
        print(f'MAPE: {mape}')

        # Now plot the training and validation loss
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        loss_plot = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # Plot Open vs Close prices
        plt.figure(figsize=(15, 5))
        plt.plot(stock_data.index, stock_data['Open'], label='Open', color='blue')
        plt.plot(stock_data.index, stock_data['Close'], label='Close', color='orange')
        plt.title('Open and Close Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['open_close'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # Plotting Volume of Stocks
        plt.figure(figsize=(15, 5))
        plt.bar(stock_data.index, stock_data['Volume'], color='purple')
        plt.title('Stock Volume')
        plt.xlabel('Date')
        plt.ylabel('Volume')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['volume'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # High and Low prices
        plt.figure(figsize=(15, 5))
        plt.plot(stock_data.index, stock_data['High'], label='High', color='green')
        plt.plot(stock_data.index, stock_data['Low'], label='Low', color='red')
        plt.title('High and Low Prices')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['high_low'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # Simple Exponential Smoothing
        x = stock_data[['Close']].values
        train_split = 0.8
        split_idx = int(np.ceil(len(x) * train_split))
        train, test = x[:split_idx], x[split_idx:]

        test_concat = np.empty((0, 1))
        for i in range(len(test)):
            train_fit = np.concatenate((train, np.asarray(test_concat)), axis=0)
            fit = SimpleExpSmoothing(np.asarray(train_fit)).fit(smoothing_level=0.2, optimized=False)
            test_pred = fit.forecast(1)
            test_concat = np.concatenate((np.asarray(test_concat), test_pred.reshape((-1, 1))))

        plt.figure(figsize=(10, 5))
        plt.plot(test, label='Actual')
        plt.plot(test_concat, label='Predicted', color='orange')
        plt.title('Actual vs Predicted Values (Simple Exponential Smoothing)')
        plt.xlabel('Time Steps')
        plt.ylabel('Values')
        plt.legend()
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['simple_exponential'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # Strip Plot
        plt.figure(figsize=(15, 5))
        sns.stripplot(data=stock_data, x=stock_data.index, y='Close', color='blue', jitter=True)
        plt.title('Strip Plot of Closing Prices')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['strip_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # Swarm Plot
        plt.figure(figsize=(15, 5))
        sns.swarmplot(data=stock_data, x=stock_data.index, y='Close', color='orange')
        plt.title('Swarm Plot of Closing Prices')
        plt.xlabel('Date')
        plt.ylabel('Close Price')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['swarm_plot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # Heatmap of correlation
        plt.figure(figsize=(10, 8))
        sns.heatmap(stock_data.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Heatmap of Stock Data Correlations')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['heatmap'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        # Candlestick chart
        buf = BytesIO()
        mpf.plot(stock_data, type='candle', style='charles', title='Candlestick Chart', ylabel='Price', savefig=buf)
        buf.seek(0)
        plots['candle'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Last 100 data points with colored candle stick
        last_100_data = stock_data[-100:]
        buf = BytesIO()
        mpf.plot(last_100_data, type='candle', style='charles', title='Last 100 Data Points (Candlestick)', ylabel='Price', savefig=buf)
        buf.seek(0)
        plots['last_100_candle'] = base64.b64encode(buf.getvalue()).decode('utf-8')

        # Boxplot of closing prices
        plt.figure(figsize=(15, 5))
        sns.boxplot(data=stock_data, y='Close', color='purple')
        plt.title('Boxplot of Closing Prices')
        plt.ylabel('Close Price')
        buf = BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plots['boxplot'] = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

    except Exception as e:
        print(f"Error: {e}")

    descriptions = {
        'loss_plot': 'This plot shows the training and validation loss over epochs, helping to understand the model\'s performance during training.',
        'open_close': 'This plot compares the opening and closing prices of the stock, illustrating market behavior over time.',
        'volume': 'This bar chart represents the stock volume traded over time, indicating market activity.',
        'high_low': 'This plot shows the high and low prices of the stock, providing insights into price fluctuations.',
        'simple_exponential': 'This plot compares the actual closing prices with predicted values using Simple Exponential Smoothing.',
        'strip_plot': 'This strip plot visualizes the distribution of closing prices, showing individual data points for clarity.',
        'swarm_plot': 'This swarm plot provides a detailed view of the closing prices, highlighting the distribution of values.',
        'heatmap': 'This heatmap illustrates the correlation between different features in the dataset, indicating relationships.',
        'candle': 'This candlestick chart visualizes the stock price movements, including open, high, low, and close prices.',
        'last_100_candle': 'This candlestick chart focuses on the last 100 data points for detailed price movement analysis.',
        'boxplot': 'This boxplot provides a summary of closing prices, highlighting median, quartiles, and potential outliers.',
    }

    return render_template('data.html', plots=plots, descriptions=descriptions, first_data=first_data, rmse=rmse, mape=mape)

if __name__ == '__main__':
    app.run(debug=True)
