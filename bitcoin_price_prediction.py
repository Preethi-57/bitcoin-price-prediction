import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor

# Load dataset
df = pd.read_csv('Dataset.csv')

# Convert Date column to datetime and sort
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')

# Rename columns for consistency
df.rename(columns={
    'Unix': 'Unix Timestamp',
    'Volume BTC': 'Volume (Crypto)',
    'Volume USD': 'Volume Base Ccy'
}, inplace=True)

# Feature engineering: date parts
df['day'] = df['Date'].dt.day
df['month'] = df['Date'].dt.month
df['year'] = df['Date'].dt.year
df['day_of_week'] = df['Date'].dt.dayofweek

# Moving averages (trend indicators)
df['ma7'] = df['Close'].rolling(window=7).mean()
df['ma30'] = df['Close'].rolling(window=30).mean()
df['ma90'] = df['Close'].rolling(window=90).mean()

# Drop rows with NaN (from rolling averages)
df.dropna(inplace=True)

# Features and target
features = [
    'Open', 'High', 'Low', 
    'Volume (Crypto)', 'Volume Base Ccy',
    'day', 'month', 'year', 'day_of_week',
    'ma7', 'ma30', 'ma90'
]
target = 'Close'

X = df[features]
y = df[target]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Initialize models
rf = RandomForestRegressor(n_estimators=150, random_state=42)
xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42)
lr = LinearRegression()

# Train models
rf.fit(X_train, y_train)
xgb.fit(X_train, y_train)
lr.fit(X_train, y_train)

# Predictions
rf_pred = rf.predict(X_test)
xgb_pred = xgb.predict(X_test)
lr_pred = lr.predict(X_test)

# Ensemble prediction (average)
ensemble_pred = (rf_pred + xgb_pred + lr_pred) / 3

# Evaluation
r2 = r2_score(y_test, ensemble_pred)
mae = mean_absolute_error(y_test, ensemble_pred)

print(f'R² Score: {r2*100:.2f}%')
print(f'Mean Absolute Error: {mae:.2f}')

# Save predictions
pred_df = pd.DataFrame({'Actual': y_test, 'Predicted': ensemble_pred})
pred_df.to_csv('data/predictions.csv', index=False)

print('\nPredictions saved to data/predictions.csv')
