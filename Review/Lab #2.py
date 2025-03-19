#%%
import pandas as pd
import yfinance as yf

#%%

stocks = ['AAPL', 'ORCL', 'TSLA', 'IBM', 'YELP', 'MSFT']

start = '2000-01-01'
end = '2023-08-29'

df = yf.download(stocks, start=start, end=end, auto_adjust=False)
print(df.tail().to_string())

#%%
for stock in enumerate(stocks):
    df['High'][stocks].plot(subplots=True,
                            xlabel = 'Date',
                            ylabel = 'High Price USD ($)',
                            sharex =  False,
                            color = '#1f77b4',
                            grid = True,
                            fontsize = 15,
                            linewidth=3,
                            figsize=(16, 8),
                            layout=(3, 2),
                            legend = False,
                            title=[f'High price history of {stock}' for stock in stocks])

plt.tight_layout()
plt.show()