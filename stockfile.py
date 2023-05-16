import pandas as pd
import yfinance as yf

df = yf.download('005930.KS', '2022-05-01', '2023-05-01')
df.to_excel("Samsung_ele.xlsx")
df.to_csv("Samsung_ele.csv")

pd.options.display.max_rows = 10
print(df)
