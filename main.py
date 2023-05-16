import numpy as np
import pandas as pd
import yfinance as yf

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

pd.options.display.max_rows = 10

# making the data
# df = yf.download('005930.KS', '2022-05-02', '2023-05-02')
# df.to_excel("Samsung_ele.xlsx")
# df.to_csv("Samsung_ele.csv")

# data loading
stock_data = pd.read_csv('Samsung_ele.csv', encoding='utf-8')
row_count, column_count = stock_data.shape
print(stock_data)

# convert error
stock_data.drop(['Date'], axis=1, inplace=True)

# set the variables
X = stock_data.drop(['Close'], axis=1)
y = stock_data['Close']

# spliting datas
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# GradientBoostingRegressor - Making, Learning
n_estimators = 500  # 결정 트리 개수 조정
gb = GradientBoostingRegressor(n_estimators=n_estimators, random_state=42)
gb.fit(X_train, y_train)

# Prediction
y_pred = gb.predict(X_test)

# Visualizing
result = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(result)

# Evaluating MSE and R-squared
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

#
print("MSE:", mse)

# Accuracy - value : 0 ~ 1
print("Accuracy (R-squared) :", r2)

