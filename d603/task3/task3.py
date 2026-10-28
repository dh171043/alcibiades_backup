import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.signal import periodogram
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

pd.set_option('display.max_columns', None) # view all columns when printing
spacer = "*"*20
def header_txt(message):
    output = f"\n{spacer} {message} {spacer}\n"
    print(output)


path = "C:/CraneDev/python_projects/wgu/d603/task3/"
# path = "D:/programming_workspace/WGU/d603/task3/"
file = "churn_clean.csv"
df = pd.read_csv(f"{path}{file}")
print(df.head())


# origial data is daily
daily_data = df

# divide by 7 to group by weeks
df['Week'] = (df['Day'] - 1) // 7
weekly_data = df.groupby('Week')['Revenue'].sum().reset_index()
weekly_data = weekly_data.iloc[:-1] #remove the partial week as that would throw off the analysis

# divide by 30 to get a decent approximation of months
df['Month'] = (df['Day'] - 1) // 30
monthly_data = df.groupby('Month')['Revenue'].sum().reset_index()
monthly_data = monthly_data.iloc[:-1] #remove the partial month as that would throw off the analysis

header_txt("Construct Subplots")
fig, axes = plt.subplots(3, 1, figsize=(11, 11))
fig.suptitle('Revenue at Different Time Granularities', fontsize=20, fontweight='bold')


# daily plot
axes[0].plot(daily_data['Day'], daily_data['Revenue'], label='Daily Revenue', color='blue')
axes[0].set_title('Daily View', fontsize=14)
axes[0].set_ylabel('Revenue ($)', fontsize=12)
axes[0].set_xlabel('Day Number', fontsize=12)
axes[0].grid(True, linestyle='--', linewidth=0.5)
axes[0].legend()


# weekly plot
axes[1].plot(weekly_data['Week'], weekly_data['Revenue'], label='Weekly Revenue', color='magenta')
axes[1].set_title('Weekly View', fontsize=14)
axes[1].set_ylabel('Total Weekly Revenue ($)', fontsize=12)
axes[1].set_xlabel('Week Number', fontsize=12)
axes[1].grid(True, linestyle='--', linewidth=0.5)
axes[1].legend()


# monthly plot
axes[2].plot(monthly_data['Month'], monthly_data['Revenue'], label='Monthly Revenue', color='green')
axes[2].set_title('Monthly View (30-day periods)', fontsize=14)
axes[2].set_ylabel('Total Monthly Revenue ($)', fontsize=12)
axes[2].set_xlabel('Month Number (30-day period)', fontsize=12)
axes[2].grid(True, linestyle='--', linewidth=0.5)
axes[2].legend()


# formatting initial realizations
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{path}time_step_realization")



# Stationary Series Plot
header_txt("Plotting Change in Revenues")
daily_data['revenue_diff'] = daily_data['Revenue'].diff()
daily_data_diff = daily_data.dropna() # the first row will be empty because of the .diff()

weekly_data['revenue_diff'] = weekly_data['Revenue'].diff()
weekly_data_diff = weekly_data.dropna() # the first row will be empty because of the .diff()

monthly_data['revenue_diff'] = monthly_data['Revenue'].diff()
monthly_data_diff = monthly_data.dropna() # the first row will be empty because of the .diff()

# construct plot
fig, axes = plt.subplots(2, 3, figsize = (20, 10))
title = "Comparing Original and Differenced Series"
fig.suptitle(title, fontsize = 18, fontweight = 'bold')

# original data
axes[0, 0].plot(daily_data.index, daily_data['Revenue'], label = 'Original Daily Data', color = 'magenta')
axes[0, 0].set_title("Original Daily Series with Trend", fontsize = 14)
axes[0, 0].set_ylabel("Total Daily Revenue ($)", fontsize = 12)
axes[0, 0].grid(True, linestyle = '--', linewidth = 0.5)
axes[0, 0].legend()

axes[0, 1].plot(weekly_data.index, weekly_data['Revenue'], label = 'Original Weekly Data', color = 'magenta')
axes[0, 1].set_title("Original Weekly Series with Trend", fontsize = 14)
axes[0, 1].set_ylabel("Total Weekly Revenue ($)", fontsize = 12)
axes[0, 1].grid(True, linestyle = '--', linewidth = 0.5)
axes[0, 1].legend()

axes[0, 2].plot(monthly_data.index, monthly_data['Revenue'], label = 'Original Monthly Data', color = 'magenta')
axes[0, 2].set_title("Original Monthly Series with Trend", fontsize = 14)
axes[0, 2].set_ylabel("Total Monthly Revenue ($)", fontsize = 12)
axes[0, 2].grid(True, linestyle = '--', linewidth = 0.5)
axes[0, 2].legend()



# differenced data
axes[1, 0].plot(daily_data_diff.index, daily_data_diff['revenue_diff'], label = 'Diffrenced Daily Revenue', color = 'green')
axes[1, 0].set_title("Differenced Daily Series (Stationary)", fontsize = 14)
axes[1, 0].set_ylabel("Change in Daily Revenue ($)", fontsize = 12)
axes[1, 0].set_xlabel("Day Number", fontsize = 8)
axes[1, 0].grid(True, linestyle = '--', linewidth = 0.5)
axes[1, 0].legend()

axes[1, 1].plot(weekly_data_diff.index, weekly_data_diff['revenue_diff'], label = 'Diffrenced Weekly Revenue', color = 'green')
axes[1, 1].set_title("Differenced Weekly Series (Stationary)", fontsize = 14)
axes[1, 1].set_ylabel("Change in Weekly Revenue ($)", fontsize = 12)
axes[1, 1].set_xlabel("Week Number", fontsize = 12)
axes[1, 1].grid(True, linestyle = '--', linewidth = 0.5)
axes[1, 1].legend()

axes[1, 2].plot(monthly_data_diff.index, monthly_data_diff['revenue_diff'], label = 'Diffrenced Monthly Revenue', color = 'green')
axes[1, 2].set_title("Differenced Monthly Series (Stationary)", fontsize = 14)
axes[1, 2].set_ylabel("Change in Monthly Revenue ($)", fontsize = 12)
axes[1, 2].set_xlabel("Month Number", fontsize = 12)
axes[1, 2].grid(True, linestyle = '--', linewidth = 0.5)
axes[1, 2].legend()


plt.tight_layout(rect = [0,0,1,0.95])
plt.savefig(f"{path}{title}")


# split the data for analysis
header_txt("test/train split")
split_point = int(len(weekly_data) * 0.8)

train = weekly_data.iloc[:split_point]
test = weekly_data.iloc[split_point:]

train.to_csv(f"{path}train.csv")
test.to_csv(f"{path}test.csv")

# verify split
header_txt("Verifying Split")
print(f"Original Shape: {weekly_data.shape}")
print(f"Train shape:    {train.shape}")
print(f"Test shape:     {test.shape}")
print(f"\n Last day of Training")
print(train.tail(1))
print(f"First day of Training")
print(train.head(1))

title = "Train and Test Split for Time Series Data"
plt.figure(figsize=(15, 7))
plt.plot(train.index, train['Revenue'], label='Training Data')
plt.plot(test.index, test['Revenue'], label='Testing Data', color='orange')
plt.title(f'{title}', fontsize=16)
plt.xlabel('Week Number')
plt.ylabel('Total Weekly Revenue ($)')
plt.legend()
plt.grid(True, linestyle='--')
plt.savefig(f"{path}{title}")


header_txt("Decomposition Graph")
decomposition = seasonal_decompose(weekly_data['Revenue'], model= 'additive', period = 52)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

fig, axes = plt.subplots(4, 1, figsize = (15, 12), sharex= True)
title = "Classical Time Series Decomposition"
fig.suptitle(f"{title}", fontsize = 20, fontweight = 'bold')

# original
axes[0].plot(weekly_data['Revenue'], label='Original Series')
axes[0].set_ylabel('Revenue')
axes[0].legend()

# trend
axes[1].plot(trend, label='Trend Component', color='blue')
axes[1].set_ylabel('Trend')
axes[1].legend()

# seasonal
axes[2].plot(seasonal, label='Seasonal Component', color='green')
axes[2].set_ylabel('Seasonal')
axes[2].legend()

# residual
axes[3].plot(residual, label='Residual (Noise)', color='red')
axes[3].set_ylabel('Residual')
axes[3].set_xlabel('Week Number')
axes[3].legend()

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{path}{title}")


# SPECTRAL DENSITY
header_txt("Spectral Density")

frequencies, density = periodogram(weekly_data['Revenue'], fs = 1)

# visualization
plt.figure(figsize=(12, 6))
plt.plot(frequencies, density)
plt.title('Spectral Density Plot', fontsize=16)
plt.xlabel('Frequency')
plt.ylabel('Power Spectral Density')
plt.grid(True)
plt.savefig(f"{path}spectral density")

# strongest peak
peak_frequency = frequencies[density.argmax()]
print(f"The strongest frequency peak is at: {peak_frequency:.4f} cycles per week")
# period is the inverse of the frequency
peak_period = 1 / peak_frequency
print(f"This corresponds to a period of: {peak_period:.2f} weeks")


# autocorrelation (ACF)
header_txt("Autocorrelation Vizualization")



fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex = True)
title = "Autocorrelation Functions"
fig.suptitle(f'{title}: Original vs. Differenced Series', fontsize=18, fontweight='bold')
plot_acf(weekly_data['Revenue'], lags=52, ax=axes[0]) # Show lags up to one year (52 weeks)
axes[0].set_title('Autocorrelation Function (ACF) for Original Weekly Revenue')
axes[0].set_ylabel('Correlation Coefficient')

plot_acf(weekly_data_diff['revenue_diff'], lags=52, ax=axes[1])
axes[1].set_title('ACF for Differenced (Stationary) Weekly Revenue')
axes[1].set_xlabel('Lag (in weeks)')
axes[1].set_ylabel('Correlation Coefficient')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{path}{title}")


# ARIMA MODEL
header_txt("Train and test ARIMA model")
train_diff = train['Revenue'].diff().dropna()
drift = train_diff.mean()
print(f"Drift of training data is: {drift:.4f}")

title = "ACF and PACF of Differenced Series"
fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle(f'{title}', fontsize=18, fontweight='bold')
# ACF
plot_acf(train_diff, lags=40, ax=axes[0])
axes[0].set_title('ACF of Differenced Series (q)')

# PACF
plot_pacf(train_diff, lags=40, ax=axes[1])
axes[1].set_title('PACF of Differenced Series (p)')

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(f"{path}{title}")

# ARIMA model parameters based off of CI test
p, d, q = 1, 1, 3

my_test = [1, 1, 3]
simple_exponential = [0, 1, 1]
default = [1, 1, 0]
very_simple = [0, 1, 3]

to_test = [my_test, simple_exponential, default, very_simple]

best_aic = float('inf')
best_model_summary = None

def test_model(arima_model, p, d, q, train = train):
    model = ARIMA(train['Revenue'], order = (p, d, q)).fit()
    current_aic = model.aic
    return model, current_aic

for i in to_test:
    header_txt(f"Model Summary for {i}")
    current_model, current_aic = test_model(i, i[0], i[1], i[2])
    if current_aic < best_aic:
        best_aic = current_aic
        best_model = current_model
        print("NEW BEST!")
        print(best_model.summary())

start_index = test.index[0]
end_index = test.index[-1]

predictions = best_model.predict(start = start_index, end = end_index)

plt.figure(figsize = (18, 10))
plt.plot(train.index, train['Revenue'], label = 'Training Data', color = 'blue')
plt.plot(test.index, test['Revenue'], label = 'Actual TestData', color = 'magenta')
plt.plot(predictions.index, test['Revenue'], label = "ARIMA Predictions", color = 'green')

title = "ARIMA Model Validation"
plt.title(f"{title}")
plt.xlabel("Week Number")
plt.ylabel("Total Weekly Revenue")
plt.legend(fontsize = 12)
plt.grid(True)
plt.savefig(f"{path}{title}")

header_txt("FORECAST")
full_model = ARIMA(weekly_data['Revenue'], order = very_simple).fit()
n_periods_to_forecast = 12

forecast = full_model.get_forecast(steps = n_periods_to_forecast)
mean_forecast = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

plt.figure(figsize = (18, 10))
plt.plot(weekly_data.index, weekly_data['Revenue'], label = 'Historical Data', color = 'blue')
plt.plot(mean_forecast.index, mean_forecast.values, label = 'Forecast Data', color = 'magenta')

plt.fill_between(
    confidence_intervals.index,
    confidence_intervals.iloc[:, 0],
    confidence_intervals.iloc[:, 1],
    color = 'green',
    alpha = .2,
    label = '95% CI'
)

title = "ARIMA Model Forecast"
plt.title(f"{title}")
plt.xlabel("Week Number")
plt.ylabel("Total Weekly Revenue")
plt.legend(fontsize = 12)
plt.grid(True)
plt.savefig(f"{path}{title}")

forecast_summary = pd.concat([mean_forecast, confidence_intervals], axis = 1)
forecast_summary.columns = ['Forecasted_Revenue', "Lower_95_CI", "Upper_95_CI"]
forecast_summary.index.name = 'Week'

forecast_summary.to_csv(f"{path}forecast_summary.csv")