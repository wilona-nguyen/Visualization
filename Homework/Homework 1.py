#%%

import yfinance as yf
from prettytable import PrettyTable

#%%
stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
start_date = '2013-01-01'
end_date = '2024-05-23'

df = yf.download(stocks, start=start_date, end=end_date, auto_adjust=False, actions=False )
print(df.tail().to_string())

#%%
features = ["High", "Low", "Open", "Close", "Volume", "Adj Close"]
col_names = ["Name", "High ($)", "Low ($)", "Open ($)", "Close ($)", "Volume", "Adj Close ($)"]

table = PrettyTable()
table.field_names = col_names

for stock in stocks:
    row = [stock]
    for feature in features:
        mean_value = df[feature][stock].mean()
        row.append(round(mean_value, 2))
    table.add_row(row)


max_values = []
min_values = []
max_companies = []
min_companies = []

for feature in features:
    max_value = df[feature].mean().max()
    min_value = df[feature].mean().min()
    max_company = df[feature].mean().idxmax()
    min_company = df[feature].mean().idxmin()

    max_values.append(round(max_value, 2))
    min_values.append(round(min_value, 2))
    max_companies.append(max_company)
    min_companies.append(min_company)


table.add_row(["Maximum Value", *max_values])
table.add_row(["Minimum Value", *min_values])
table.add_row(["Maximum Company Name", *max_companies])
table.add_row(["Minimum Company Name", *min_companies])

print(table.get_string(title="Mean Value Comparison"))


#%%
var_table = PrettyTable()
var_table.field_names = col_names

for stock in stocks:
    row = [stock]
    for feature in features:
        var_value = df[feature][stock].var()
        row.append(round(var_value, 2))
    var_table.add_row(row)


max_values = []
min_values = []
max_companies = []
min_companies = []

for feature in features:
    max_value = df[feature].var().max()
    min_value = df[feature].var().min()
    max_company = df[feature].var().idxmax()
    min_company = df[feature].var().idxmin()

    max_values.append(round(max_value, 2))
    min_values.append(round(min_value, 2))
    max_companies.append(max_company)
    min_companies.append(min_company)


var_table.add_row(["Maximum Value", *max_values])
var_table.add_row(["Minimum Value", *min_values])
var_table.add_row(["Maximum Company Name", *max_companies])
var_table.add_row(["Minimum Company Name", *min_companies])

print(var_table.get_string(title="Variance Comparison"))


#%%

std_table = PrettyTable()
std_table.field_names = col_names

for stock in stocks:
    row = [stock]
    for feature in features:
        std_value = df[feature][stock].std()
        row.append(round(std_value, 2))
    std_table.add_row(row)


max_values = []
min_values = []
max_companies = []
min_companies = []

for feature in features:
    max_value = df[feature].std().max()
    min_value = df[feature].std().min()
    max_company = df[feature].std().idxmax()
    min_company = df[feature].std().idxmin()

    max_values.append(round(max_value, 2))
    min_values.append(round(min_value, 2))
    max_companies.append(max_company)
    min_companies.append(min_company)


std_table.add_row(["Maximum Value", *max_values])
std_table.add_row(["Minimum Value", *min_values])
std_table.add_row(["Maximum Company Name", *max_companies])
std_table.add_row(["Minimum Company Name", *min_companies])

print(std_table.get_string(title="Standard Deviation Value Comparison"))

#%%

median_table = PrettyTable()
median_table.field_names = col_names

for stock in stocks:
    row = [stock]
    for feature in features:
        median_value = df[feature][stock].median()
        row.append(round(median_value, 2))
    median_table.add_row(row)


max_values = []
min_values = []
max_companies = []
min_companies = []

for feature in features:
    max_value = df[feature].median().max()
    min_value = df[feature].median().min()
    max_company = df[feature].median().idxmax()
    min_company = df[feature].median().idxmin()

    max_values.append(round(max_value, 2))
    min_values.append(round(min_value, 2))
    max_companies.append(max_company)
    min_companies.append(min_company)


median_table.add_row(["Maximum Value", *max_values])
median_table.add_row(["Minimum Value", *min_values])
median_table.add_row(["Maximum Company Name", *max_companies])
median_table.add_row(["Minimum Company Name", *min_companies])

print(median_table.get_string(title="Median Value Comparison"))


#%%
for stock in stocks:

    stock_columns = [col for col in df.columns if stock in col]
    stock_df = df[stock_columns]


    stock_correlation_matrix = stock_df.corr()


    print(f"{stock} Correlation Matrix:")
    print(stock_correlation_matrix)
    print("\n" + "=" * 50 + "\n")


#%%
#8

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



