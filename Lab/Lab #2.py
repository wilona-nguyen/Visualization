import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf


#%%

stocks = ['AAPL','ORCL', 'TSLA', 'IBM','YELP', 'MSFT']
start_date = '2000-01-01'
end_date = '2023-08-29'

df = yf.download(stocks, start=start_date, end=end_date, auto_adjust=False, actions=False )
#print(df.tail().to_string())

#%%
#Q1 - "High"

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


#%%
#Q2 - "Low"

for stock in enumerate(stocks):
    df['Low'][stocks].plot(subplots=True,
                            xlabel = 'Date',
                            ylabel = 'Low Price USD ($)',
                            sharex =  False,
                            color = '#1f77b4',
                            grid = True,
                            fontsize = 15,
                            linewidth=3,
                            figsize=(16, 8),
                            layout=(3, 2),
                            legend = False,
                            title=[f'Low price history of {stock}' for stock in stocks])

plt.tight_layout()
plt.show()

#%%
#Q2 - "Open"

for stock in enumerate(stocks):
    df['Open'][stocks].plot(subplots=True,
                            xlabel = 'Date',
                            ylabel = 'Open Price USD ($)',
                            sharex =  False,
                            color = '#1f77b4',
                            grid = True,
                            fontsize = 15,
                            linewidth=3,
                            figsize=(16, 8),
                            layout=(3, 2),
                            legend = False,
                            title=[f'Open price history of {stock}' for stock in stocks])

plt.tight_layout()
plt.show()

#%%
#Q2 - "Close"

for stock in enumerate(stocks):
    df['Close'][stocks].plot(subplots=True,
                            xlabel = 'Date',
                            ylabel = 'Close Price USD ($)',
                            sharex =  False,
                            color = '#1f77b4',
                            grid = True,
                            fontsize = 15,
                            linewidth=3,
                            figsize=(16, 8),
                            layout=(3, 2),
                            legend = False,
                            title=[f'Close price history of {stock}' for stock in stocks])

plt.tight_layout()
plt.show()

#%%
#Q2 - "Volume"

for stock in enumerate(stocks):
    df['Volume'][stocks].plot(subplots=True,
                            xlabel = 'Date',
                            ylabel = 'Volume',
                            sharex =  False,
                            color = '#1f77b4',
                            grid = True,
                            fontsize = 15,
                            linewidth=3,
                            figsize=(16, 8),
                            layout=(3, 2),
                            legend = False,
                            title=[f'Volume history of {stock}' for stock in stocks])

plt.tight_layout()
plt.show()

#%%
#Q2 - "Adj Close"

for stock in enumerate(stocks):
    df['Adj Close'][stocks].plot(subplots=True,
                            xlabel = 'Date',
                            ylabel = 'Adj Close Price USD ($)',
                            sharex =  False,
                            color = '#1f77b4',
                            grid = True,
                            fontsize = 15,
                            linewidth=3,
                            figsize=(16, 8),
                            layout=(3, 2),
                            legend = False,
                            title=[f'Adj Close price history of {stock}' for stock in stocks])

plt.tight_layout()
plt.show()

#%%
#Q3
df['High'][stocks].plot(kind='hist',
                        bins=50,
                        subplots=True,
                        xlabel='Value in USD ($)',
                        ylabel='Frequency',
                        sharex=False,
                        sharey = False,
                        color='#1f77b4',
                        grid=True,
                        fontsize=15,
                        figsize=(16, 8),
                        layout=(3, 2),
                        legend=False,
                        title=[f'High price history of {stock}' for stock in stocks])

plt.tight_layout()
plt.show()

#%%
#Q4 - "Low"

df['Low'][stocks].plot(kind='hist',
                        bins=50,
                        subplots=True,
                        xlabel='Value in USD ($)',
                        ylabel='Frequency',
                        sharex=False,
                        sharey = False,
                        color='#1f77b4',
                        grid=True,
                        fontsize=15,
                        figsize=(16, 8),
                        layout=(3, 2),
                        legend=False,
                        title=[f'Low price history of {stock}' for stock in stocks])

plt.tight_layout()
plt.show()

#%%
#Q4 - "Open"

df['Open'][stocks].plot(kind='hist',
                        bins=50,
                        subplots=True,
                        xlabel='Value in USD ($)',
                        ylabel='Frequency',
                        sharex=False,
                        sharey = False,
                        color='#1f77b4',
                        grid=True,
                        fontsize=15,
                        figsize=(16, 8),
                        layout=(3, 2),
                        legend=False,
                        title=[f'Open price history of {stock}' for stock in stocks])

plt.tight_layout()
plt.show()

#%%
#Q4 - "Close"

df['Close'][stocks].plot(kind='hist',
                        bins=50,
                        subplots=True,
                        xlabel='Value in USD ($)',
                        ylabel='Frequency',
                        sharex=False,
                        sharey = False,
                        color='#1f77b4',
                        grid=True,
                        fontsize=15,
                        figsize=(16, 8),
                        layout=(3, 2),
                        legend=False,
                        title=[f'Close price history of {stock}' for stock in stocks])

plt.tight_layout()
plt.show()

#%%
#Q4 - "Volume"

df['Volume'][stocks].plot(kind='hist',
                        bins=50,
                        subplots=True,
                        xlabel='Shares',
                        ylabel='Frequency',
                        sharex=False,
                        sharey = False,
                        color='#1f77b4',
                        grid=True,
                        fontsize=15,
                        figsize=(16, 8),
                        layout=(3, 2),
                        legend=False,
                        title=[f'Volume history of {stock}' for stock in stocks])

plt.tight_layout()
plt.show()

#%%
#Q4 - "Adj Close"

df['Adj Close'][stocks].plot(kind='hist',
                        bins=50,
                        subplots=True,
                        xlabel='Value in USD ($)',
                        ylabel='Frequency',
                        sharex=False,
                        sharey = False,
                        color='#1f77b4',
                        grid=True,
                        fontsize=15,
                        figsize=(16, 8),
                        layout=(3, 2),
                        legend=False,
                        title=[f'Adj Close price history of {stock}' for stock in stocks])

plt.tight_layout()
plt.show()

#%%
#Q5

pd.plotting.scatter_matrix(df.xs('AAPL', axis=1, level=1),
                           hist_kwds={'bins': 50},
                           alpha=0.5,
                           s=10,
                           diagonal='kde',
                           figsize=(10, 10))

plt.suptitle('Scatter Matrix of AAPL', fontsize=15)
plt.show()

#%%
#Q6
for stock in stocks:
    # Access data for each stock
    pd.plotting.scatter_matrix(df.xs(stock, axis=1, level=1),
                           hist_kwds={'bins': 50},
                           alpha=0.5,
                           s=10,
                           diagonal='kde',
                           figsize=(10, 10))

    plt.suptitle(f'Scatter Matrix of {stock}', fontsize=15)
    plt.show()
