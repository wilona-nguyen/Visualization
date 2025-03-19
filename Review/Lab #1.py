#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

np.random.seed(6401)
#%% - 1
np.random.seed(6401)

mean_x = 0
var_x = 1

mean_y = 5
var_y = 2

obs = 1000

x = np.random.normal(mean_x, np.sqrt(var_x), obs)
y = np.random.normal(mean_y, np.sqrt(var_y), obs)

#%% - 2
def pearson_correlation(x, y):
    mean_x, mean_y = x.mean(), y.mean()
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x) ** 2) * np.sum((y - mean_y) ** 2))
    result = numerator / denominator
    return round(result, 2)

pearson_correlation(x, y)

#%% - 3
print(f"The sample mean of random variable x is: {x.mean(): .2f}")
print(f"The sample mean of random variable y is: {y.mean(): .2f}")
print(f"The sample variance of random variable x is: {x.var(): .2f}")
print(f"The sample variance of random variable y is: {y.var(): .2f}")

#%% - 4
plt.plot(x)
plt.plot(y)

plt.xlabel("Number of Observations")
plt.ylabel("Value")
plt.title("Distribution of Random Variables")
plt.legend(["Random variable x", "Random variable y"])
plt.show()

#%% - 5
plt.hist(x, bins=100)
plt.hist(y, bins=100)

plt.xlabel("Number of Observations")
plt.ylabel("Frequency")
plt.title("Histogram of Random Variables")
plt.legend(["Random variable x", "Random variable y"])
plt.show()

#%% - 6
url = "https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/tute1.csv"
df = pd.read_csv(url)

print(df.tail(5))

#%% - 7
sales_ad_cor = pearson_correlation(df['Sales'], df['AdBudget'])
sales_gdp_cor = pearson_correlation(df['Sales'], df['GDP'])
ad_gdp_cor = pearson_correlation(df['AdBudget'], df['GDP'])

print(f"The sample Pearson's correlation coefficient between Sale & AdBudget is: {pearson_correlation(df['Sales'], df['AdBudget']): .2f}")
print(f"The sample Pearson's correlation coefficient between Sale & GDP is: {pearson_correlation(df['Sales'], df['GDP']): .2f}")
print(f"The sample Pearson's correlation coefficient between AdBudget & GDP is: {pearson_correlation(df['AdBudget'], df['GDP']): .2f}")

#%% - 8
plt.scatter(df['Sales'], df['AdBudget'])

plt.xlabel("Sales")
plt.ylabel("AdBudget")
plt.grid(True)
plt.title(f" {"Scatter plot of Sales and AdBudget"} (Correlation: {sales_ad_cor: .2f})")
plt.show()

#%% - 9
plt.scatter(df['GDP'], df['AdBudget'])

plt.xlabel("GDP")
plt.ylabel("AdBudget")
plt.grid(True)
plt.title(f" {"Scatterplot of GDP and AdBudget"} (Correlation: {ad_gdp_cor: .2f})")
plt.show()

#%% - 11
plt.figure(figsize=(16,8))
plt.plot(df['Date'], df['Sales'])
plt.plot(df['Date'],df['GDP'])
plt.plot(df['Date'], df['AdBudget'])

tick_pos = df['Date'][::7]

plt.xlabel("Date")
plt.ylabel("Value")
plt.xticks(tick_pos, rotation=45)
plt.title("Distribution of Sales, AdBudget and GDP")
plt.legend(["Sales", "GDP", "AdBudget"], loc="best")
plt.show()

#%% - 12
plt.hist(df['Sales'], bins=30)
plt.hist(df['AdBudget'], bins=30)
plt.hist(df['GDP'], bins=30)

plt.xlabel("Category")
plt.ylabel("Frequency")
plt.title("Distribution of Sales, AdBudget and GDP")
plt.legend(["Sales", "AdBudget", "GDP"], loc="best")
plt.show()

