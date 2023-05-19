import numpy as np
import pandas as pd
from pykrx import stock

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

import time
import matplotlib.pyplot as plt

pd.options.display.max_rows = 10

# 데이터 불러오기
start_date = '2022-05-01'
end_date = '2023-05-01'
df = stock.get_market_fundamental_by_date(start_date, end_date, '005930')
df.reset_index(inplace=True)
print(df)

# 주식 가격 데이터 가져오기
stock_data = stock.get_market_ohlcv_by_date(start_date, end_date, '005930')
stock_data.reset_index(inplace=True)
print(stock_data)

# 데이터 병합
merged_data = pd.merge(stock_data, df, on='날짜')

# 변수 설정 - Close는 주식 예측 변수 중 가장 중요하므로 종속 변수로 설정
X = merged_data[['시가', '고가', '저가', '거래량']]
y = merged_data['종가']

# 데이터 분할 (훈련 데이터와 테스트 데이터)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# GradientBoostingRegressor - 모델 생성 및 학습
# 학습률 0.01 ~ 0.1 조정
gb = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.1, random_state=42, max_depth=3)

# # 그리드 탐색을 위한 파라미터 설정, greed Algo
# param_grid = {
#     'n_estimators': [50, 100, 200, 500, 1000],  # 트리 개수
#     'learning_rate': [0.05, 0.01, 0.1, 0.5, 0.9],  # 학습률
#     'max_depth': [3, 5, 7, 21, 42]  # 트리 깊이
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

# # 최적의 파라미터와 점수 출력
# print("Best Parameters: ", grid_search.best_params_)
# print("Best Score: ", -grid_search.best_score_)
# print("grid 탐색 시간: ", cross_val_time, "초")

# 훈련 시간 측정
start_time = time.time()
gb.fit(X_train, y_train)
end_time = time.time()
training_time = end_time - start_time

# 예측
y_pre = gb.predict(X_test)
y_pre_rounded = np.round(y_pre, 1)  # 예측값을 소수 첫째 자리까지 반올림

# 시각화
result = pd.DataFrame({'실제값': y_test, '예측값': y_pre_rounded}).reset_index(drop=True)
print(result)

# MSE 및 R-squared 평가
mse_gb = mean_squared_error(y_test, y_pre)
r2_gb = r2_score(y_test, y_pre)

# MSE (gb) 출력
print("MSE (Gradient Boosting): ", mse_gb)
print("훈련 시간 (Gradient Boosting): ", training_time, "초")
print("정확도 (R-squared) (Gradient Boosting): ", r2_gb)

# 교차 검증 (gb)
start_time = time.time()
scores_gb = cross_val_score(gb, X, y, cv=10, scoring='neg_mean_squared_error')
end_time = time.time()
cross_val_time = end_time - start_time

mse_scores = -scores_gb  # 양수로 변환하여 MSE 스코어 계산

cv_results = np.round(pd.DataFrame({'Cross-Validation MSE Scores': mse_scores}), 1)
print(cv_results)

print("Cross-Validation 평균 MSE: ", mse_scores.mean())
print("교차 검증 시간: ", cross_val_time, "초")
print("\n")

# 실제 값과 예측 값 그래프
# 한글표시 x
plt.figure(figsize=(12, 6))
plt.plot(y_test.values, label='Actual')
plt.plot(y_pre, label='Predicted')
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.title('Actual vs Predicted Stock Price')
plt.legend()
plt.show()

