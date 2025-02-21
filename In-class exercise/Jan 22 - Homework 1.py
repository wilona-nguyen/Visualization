#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
#yf.pdr_override()

from prettytable import PrettyTable

#%%


name = ["AAPL"]
start_date = '2000-01-01'
end_date = '2025-01-23'

#df = data.get_data_yahoo(name[0], start=start_date, end=end_date)

df = yf.download(name[0], start=start_date, end=end_date)
print(df.tail().to_string())

df.iloc[-1,:]
df[:-1].plot()

#%%

np.random.seed(6401)

data = np.random.randn(6,4)
print(data)

df = pd.DataFrame(data,
                  columns=['A', 'B', 'C', 'D'],
                  index=np.arange(6))

x = PrettyTable()

for i in range(6):
    x.add_row(df.iloc[i,:])
x.field_names = ['A', 'B', 'C', 'D']
print(x.get_string(title="Random Data"))

#%%
np.random.seed(6401)
#exercise
N = 10000
mean = 0
var = 2

x1 = np.random.normal(mean, np.sqrt(var), N)
x2 = np.random.normal(mean+1, np.sqrt(var+1), N)
x3 = np.random.normal(mean+2, np.sqrt(var+2), N)
x4 = np.random.normal(mean+3, np.sqrt(var+3), N)
x5 = np.random.normal(mean+4, np.sqrt(var+4), N)



X = np.vstack((x1, x2, x3, x4, x5)).T
df = pd.DataFrame(X)

cov = (df - df.mean()).T @ (df -df.mean()) / (N-1)
print(cov) #5 features are independetn from each other --> near 0; diagnal makes sense


