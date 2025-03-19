#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from prettytable import PrettyTable

#%%
stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']

start = '2013-01-01'
end = '2024-05-23'

df = yf.download(stocks, start=start, end=end, auto_adjust=False)
print(df.tail().to_string())

#%% - 2-5. Create table for mean, variance, std, median

features = ['High', 'Low', 'Open', 'Close', 'Volume', 'Adj Close']
col_names = ['Name', 'High ($)', 'Low ($)', 'Open ($)', 'Close ($)', 'Volume ($)', 'Adj Close ($)']

mean_table = PrettyTable(col_names)


for stock in stocks:
    row = [stock]
    for feature in features:
        mean_value = df[feature][stock].mean()
        row.append(round(mean_value, 2))
    mean_table.add_row(row)

min_values = []
max_values = []
min_companies = []
max_companies = []

for feature in features:
    max_value = df[feature].mean().max()
    min_value = df[feature].mean().min()
    max_company = df[feature].mean().idxmax()
    min_company = df[feature].mean().idxmin()

    max_values.append(round(max_value, 2))
    min_values.append(round(min_value, 2))
    max_companies.append(max_company)
    min_companies.append(min_company)

mean_table.add_row(["Maximum Value", *max_values])
mean_table.add_row(["Minimum Value", *min_values])
mean_table.add_row(["Maximum Company Name", *max_companies])
mean_table.add_row(["Minimum Company Name", *min_companies])




print(mean_table.get_string(title="Mean Value Comparison"))

#%% - 6,7. Calculate the correlation matrix for all companies with all the given features
for stock in stocks:

    stock_columns = [col for col in df.columns if stock in col]
    stock_df = df[stock_columns]


    stock_correlation_matrix = stock_df.corr()


    print(f"{stock} Correlation Matrix:")
    print(round(stock_correlation_matrix, 2))
    print("\n" + "=" * 50 + "\n")

#%% - 8
risk_table = PrettyTable()
risk_table.field_names = ["Rank", "Company", "Std Dev ($)", "Mean Adj Close ($)", "Coefficient of Variation (%)", "Risk Level"]

cv_values = []
for stock in stocks:
    std_dev = df["Adj Close"][stock].std()
    mean_price = df["Adj Close"][stock].mean()
    cv = std_dev / mean_price
    cv_values.append((stock, std_dev, mean_price, cv))


cv_values.sort(key=lambda x: x[3])


for rank, (stock, std_dev, mean_price, cv) in enumerate(cv_values, start=1):
    risk = "Low Risk (Safest)" if cv == min(x[3] for x in cv_values) else \
           "High Risk (Riskiest)" if cv == max(x[3] for x in cv_values) else \
           "Medium Risk"
    risk_table.add_row([rank, stock, round(std_dev, 2), round(mean_price, 2), round(cv * 100, 2), risk])

print(risk_table.get_string(title="Stock Volatility Comparison"))



