# Developed by : Sriram S
# reg no : 212222240105
# Ex.No: 6               HOLT WINTERS METHOD
### Date: 

### AIM:
To analyze walmart sales dataset using the Holt-Winters exponential smoothing method. 

### ALGORITHM:
1.Import Libraries: Import necessary libraries for data manipulation, numerical operations, visualization, and time series analysis (pandas, numpy, matplotlib, and statsmodels).

2.Load Dataset: Load the dataset from a CSV file into a Pandas DataFrame.

3.Preprocess Date Column: Convert the 'Date' column to a datetime format and set it as the DataFrame index to facilitate time series analysis.

4.Clean 'Close' Column: Convert the 'Close' column to numeric type, removing any invalid values. Drop rows with missing values in the 'Close' column to ensure a clean dataset.

5.Extract Relevant Data: Extract the cleaned 'Close' column for time series forecasting.

6.Fit Holt-Winters Model: Apply the Holt-Winters Exponential Smoothing model to the cleaned closing price data, specifying additive trend and seasonal components along with the seasonal periods.

7.Forecast Future Values: Use the fitted model to forecast the closing prices for the next 30 business days.

8.Visualize Results: Plot the original closing price data and the forecasted values on a graph to visually compare the historical and predicted prices.
### PROGRAM:
```python
# Import the necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Load the dataset
data = pd.read_csv('/content/WMT.csv') 
data['dateTime'] = pd.to_datetime(data['Date'])  
data.set_index('dateTime', inplace=True)

print(data.columns)

data['Close'] = pd.to_numeric(data['Close'], errors='coerce')  

clean_data = data.dropna(subset=['Close'])  

close_data_clean = clean_data['Close'] 

model = ExponentialSmoothing(close_data_clean, trend="add", seasonal="add", seasonal_periods=12)
fit = model.fit()

n_steps = 30
forecast = fit.forecast(steps=n_steps)

plt.figure(figsize=(10, 6))
plt.plot(close_data_clean.index, close_data_clean, label='Original Data')
plt.plot(pd.date_range(start=close_data_clean.index[-1], periods=n_steps+1, freq='B')[1:], forecast, label='Forecast', color='orange')
plt.xlabel('Date')
plt.ylabel('Closing Price')
plt.title('Holt-Winters Forecast for Ethereum Prices')
plt.legend()
plt.show()
```
### OUTPUT:
## FINAL_PREDICTION

![image](https://github.com/user-attachments/assets/cc910d10-9664-4dcb-869d-76b69ea8a7db)

### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
