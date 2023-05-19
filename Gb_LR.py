########### 선형모델과 비선형모델의 정확도 차이 분석 #############

import numpy as np
import pandas as pd
from pykrx import stock

# Linear, Gradient Boosting
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# LSTM
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.wrappers.scikit_learn import KerasRegressor

import time
import matplotlib.pyplot as plt

pd.options.display.max_rows = 10

############################################################################################
# 제무제표 불러오기
start_date = '2022-05-01'
end_date = '2023-04-28'
df = stock.get_market_fundamental_by_date(start_date, end_date, '005930')
df.reset_index(inplace=True)
print(df)

# 주식 가격 데이터 가져오기
stock_data = stock.get_market_ohlcv_by_date(start_date, end_date, '005930')
stock_data.reset_index(inplace=True)
print(stock_data)

# 데이터 병합
merged_data = pd.merge(stock_data, df, on='날짜')

# 종가 데이터 추출
close_price = merged_data['종가'].values.reshape(-1, 1)

# 변수 설정 - Close는 주식 예측 변수 중 가장 중요하므로 종속 변수로 설정
X = merged_data[['시가', '고가', '저가', '거래량', 'EPS', 'PER']]
y = merged_data['종가']

# 데이터 분할 (훈련 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

############################################################################################
# LinearRegression - 모델 생성 및 학습
lr = LinearRegression()

# GradientBoostingRegressor - 모델 생성 및 학습
# 학습률 0.01 ~ 0.1 조정
gb = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.1, random_state=42, max_depth=3)

# 훈련 시간 측정
start_time = time.time()
lr.fit(X_train, y_train)
end_time = time.time()
training_time_lr = end_time - start_time

# 훈련 시간 측정
start_time = time.time()
gb.fit(X_train, y_train)
end_time = time.time()
training_time_gb = end_time - start_time

# 예측 gb
y_pred_gb = gb.predict(X_test)
y_pred_rounded_gb = np.round(y_pred_gb, 1)  # 예측값을 소수 첫째 자리까지 반올림

# 예측 lr
y_pred_lr = lr.predict(X_test)
y_pred_rounded_lr = np.round(y_pred_lr, 1)  # 예측값을 소수 첫째 자리까지 반올림

# 시각화 gb
result = pd.DataFrame({'실제값': y_test, '예측값': y_pred_rounded_gb}).reset_index(drop=True)
print(result)

# 시각화 lr
result = pd.DataFrame({'실제값': y_test, '예측값': y_pred_rounded_lr}).reset_index(drop=True)
print(result)

########################################################################################
# MSE 및 R-squared 평가 gb
mse_gb = mean_squared_error(y_test, y_pred_gb)
r2_gb = r2_score(y_test, y_pred_gb)

# MSE 및 R-squared 평가 lr
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# MSE (gb) 출력
print("MSE (Gradient Boosting): ", mse_gb)
print("훈련 시간 (Gradient Boosting): ", training_time_gb, "초")
print("정확도 (R-squared) (Gradient Boosting): ", r2_gb)

# MSE (lr) 출력
print("MSE (Linear Regression): ", mse_lr)
print("훈련 시간 (Linear Regression): ", training_time_lr, "초")
print("정확도 (R-squared) (Linear Regression): ", r2_lr)

###########################################################################################
# 교차 검증 (gb)
start_time = time.time()
scores_gb = cross_val_score(gb, X, y, cv=10, scoring='neg_mean_squared_error')
end_time = time.time()
cross_val_time_gb = end_time - start_time

# 교차 검증 (lr)
start_time = time.time()
scores_lr = cross_val_score(lr, X, y, cv=10, scoring='neg_mean_squared_error')
end_time = time.time()
cross_val_time_lr = end_time - start_time

mse_scores_gb = -scores_gb  # 양수로 변환하여 MSE 스코어 계산
mse_scores_lr = -scores_lr  # 양수로 변환하여 MSE 스코어 계산

cv_results_gb = np.round(pd.DataFrame({'Cross-Validation MSE Scores (Gradient Boosting)': mse_scores_gb}), 1)
print(cv_results_gb)
cv_results_lr = np.round(pd.DataFrame({'Cross-Validation MSE Scores (Linear Regression)': mse_scores_lr}), 1)
print(cv_results_lr)

print("Cross-Validation (Gradient Boosting) 평균 MSE: ", mse_scores_gb.mean())
print("교차 검증 시간: ", cross_val_time_gb, "초")
print("Cross-Validation (Linear Regression) 평균 MSE: ", mse_scores_lr.mean())
print("교차 검증 시간: ", cross_val_time_lr, "초")
print("\n")

############################################################################################
# # 그리드 탐색을 위한 파라미터 설정, greed Algo
# param_grid = {
#     'n_estimators': [3000,3500, 2500],
#     'learning_rate': [0.1, 0.15, 0.2],
#     'max_depth': [3, 5, 7, 21, 42]
# }
#
# # 그리드 탐색 객체 생성
# grid_search = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42),
#                            param_grid=param_grid,
#                            scoring='neg_mean_squared_error',
#                            cv=5)
#
# # 그리드 탐색 수행
# start_time = time.time()
# grid_search.fit(X_train, y_train)
# end_time = time.time()
# cross_val_time = end_time - start_time
#
# # 최적의 파라미터와 점수 출력
# print("Best Parameters: ", grid_search.best_params_)
# print("Best Score: ", -grid_search.best_score_)
# print("grid 탐색 시간: ", cross_val_time, "초")

############################################################################################

# 종가 비교
actual_price = y_test.values

# Linear Regression 종가 비교
lr_price = y_pred_rounded_lr

# Gradient Boosting 종가 비교
gb_price = y_pred_rounded_gb

# 종가 비교를 위한 데이터프레임 생성
comparison_df = pd.DataFrame({'Actual': actual_price, 'Linear Regression': lr_price, 'Gradient Boosting': gb_price})

# 종가를 기준으로 퍼센트 변화 계산
comparison_df['LR % Change'] = (comparison_df['Linear Regression'] - comparison_df['Actual']) / comparison_df['Actual'] * 100
comparison_df['GB % Change'] = (comparison_df['Gradient Boosting'] - comparison_df['Actual']) / comparison_df['Actual'] * 100

#############################################################################################################
print(comparison_df)
# 실제 값과 예측 값 그래프
# 한글표시안됨
# plt.figure(figsize=(12, 6))
# plt.plot(y_test.values, label='Actual')
# plt.plot(y_pred_rounded_lr, label='Linear Regression')
# plt.plot(y_pred_rounded_gb, label='Gradient Boosting')
# plt.xlabel('Index')
# plt.ylabel('Stock Price')
# plt.title('Actual vs Predicted Stock Price')
# plt.legend()
# plt.show()

fig, ax = plt.subplots(2, 1, figsize=(12, 8))

# Linear Regression
ax[0].plot(y_test.values, 'o-', label='Actual')
ax[0].plot(y_pred_rounded_lr, 'o-', label='Linear Regression')
ax[0].set_xlabel('Index')
ax[0].set_ylabel('Stock Price')
ax[0].set_title('Linear Regression: Actual vs Predicted Stock Price')
ax[0].legend()

# Gradient Boosting
ax[1].plot(y_test.values, 'o-', label='Actual')
ax[1].plot(y_pred_rounded_gb, 'o-', label='Gradient Boosting')
ax[1].set_xlabel('Index')
ax[1].set_ylabel('Stock Price')
ax[1].set_title('Gradient Boosting: Actual vs Predicted Stock Price')
ax[1].legend()

plt.tight_layout()
plt.show()
############################################################################################
# 실제 다음 일주일의 주식 종가 데이터 가져오기
next_week_end_date = pd.to_datetime(end_date) + pd.DateOffset(days=50)
next_week_df = stock.get_market_fundamental_by_date(end_date, next_week_end_date, '005930')
next_week_stock_data = stock.get_market_ohlcv_by_date(end_date, next_week_end_date, '005930')
next_week_stock_data.reset_index(inplace=True)

# 데이터 병합
next_week_merged_data = pd.merge(next_week_stock_data, next_week_df, on='날짜')

# 종가 데이터 추출
next_week_close_price = next_week_merged_data['종가'].values.reshape(-1, 1)

# 종가 데이터를 기반으로 다음 일주일 동안의 예측값 계산
next_week_features = next_week_merged_data[['시가', '고가', '저가', '거래량', 'EPS', 'PER']]
next_week_lr_prediction = lr.predict(next_week_features)
next_week_gb_prediction = gb.predict(next_week_features)

# 데이터프레임 생성
next_week_predictions = pd.DataFrame({
    '날짜': next_week_merged_data['날짜'],
    '실제 종가': next_week_close_price.flatten(),
    'Linear Regression 예측 종가': np.round(next_week_lr_prediction, 1),
    'Gradient Boosting 예측 종가': np.round(next_week_gb_prediction, 1)
})

print(next_week_predictions)

comparison_df = next_week_predictions[['날짜', '실제 종가', 'Linear Regression 예측 종가', 'Gradient Boosting 예측 종가']].copy()
comparison_df['LR-Actual(%)'] = ((comparison_df['Linear Regression 예측 종가'] - comparison_df['실제 종가']) / comparison_df['실제 종가']) * 100
comparison_df['GB-Actual(%)'] = ((comparison_df['Gradient Boosting 예측 종가'] - comparison_df['실제 종가']) / comparison_df['실제 종가']) * 100


print(comparison_df)

############################################################################################
