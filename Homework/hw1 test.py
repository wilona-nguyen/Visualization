import pandas as pd
import yfinance as yf
from prettytable import PrettyTable

stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
start_date = '2013-01-01'
end_date = '2024-05-23'

df = yf.download(stocks, start=start_date, end=end_date, auto_adjust=False, actions=False )
print(df.tail().to_string())

table = PrettyTable()
table.field_names = ["Name", "High ($)", "Low ($)", "Open ($)", "Close ($)", "Volume", "Adj Close ($)"]
for stock in stocks:
    row = [stock]
    for feature in ["High", "Low", "Open", "Close", "Volume", "Adj Close"]:
        mean_value = df[feature][stock].mean()
        row.append(round(mean_value,2))
    table.add_row(row)

max_values = []
min_values = []
max_companies = []
min_companies = []

for feature in ["High", "Low", "Open", "Close", "Volume", "Adj Close"]:
    max_value = df[feature].mean().max()
    min_value = df[feature].mean().min()
    max_company = df[feature].mean().idxmax()
    min_company = df[feature].mean().idxmin()
    max_values.append(round(max_value,2))

table.add_row(["Maximum Value", *max_values])
table.add_row(['Minimum Value', *min_values])
table.add_row(['Maximum Company Name', *max_companies])
table.add_row(['Minimum Company Name', *min_companies])

print(table.get_string(title="Mean Value Comparison"))