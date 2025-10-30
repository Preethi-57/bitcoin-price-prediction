#Bitcoin Price Prediction using Ensemble Learning

This project predicts Bitcoin closing prices using ensemble machine learning models.  
It combines XGBoost, Random Forest, and Linear Regression to increase accuracy.

# Dataset Columns
- Unix Timestamp
- Date
- Symbol
- Open
- High
- Low
- Close
- Volume BTC
- Volume USD

The script automatically engineers extra features like:
- Moving averages (ma7, ma30, ma90)
- Date-based features (day, month, year, day_of_week)

#  Steps to Run
1. Place your dataset as `Dataset.csv`.
2. Install dependencies:
   
   pip install -r requirements.txt
   
3. Run the model:
   
   python bitcoin_price_prediction.py
  
4. Results will include model accuracy (R² score) and predicted prices.

# Models Used
- RandomForestRegressor
- XGBRegressor
- LinearRegression
- Ensemble Average (for better accuracy)


