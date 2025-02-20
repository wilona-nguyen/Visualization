#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#%%


#1
mean_x = float(input("Enter the mean for the first random variable (x): "))
variance_x = float(input("Enter the variance for the first random variable (x): "))

mean_y = float(input("Enter the mean for the first random variable (y): "))
variance_y = float(input("Enter the variance for the first random variable (y): "))

n_samples = 1000

np.random.seed(6401)
x = np.random.normal(loc=mean_x, scale=np.sqrt(variance_x), size=n_samples)
y = np.random.normal(loc=mean_y, scale=np.sqrt(variance_y), size=n_samples)



#%%
#2
def pearson_correlation(x, y):
    mean_x, mean_y = x.mean(), y.mean()
    numerator = np.sum((x - mean_x) * (y - mean_y))
    denominator = np.sqrt(np.sum((x - mean_x)**2) * np.sum((y - mean_y)**2))
    result = numerator / denominator
    return round(result, 3)

print(pearson_correlation(x, y))

#%%
#3
def display_statistics(x, y):
    print(f"The sample mean of random variable x is: {x.mean():.2f}")
    print(f"The sample mean of random variable y is: {y.mean():.2f}")
    print(f"The sample variance of random variable x is: {np.var(x, ddof=1):.2f}")
    print(f"The sample variance of random variable y is: {np.var(y, ddof=1):.2f}")

display_statistics(x, y)

#%%
#4
plt.figure()
plt.plot(x, label='Random Variable X', alpha=0.7)
plt.plot(y, label='Random Variable Y', alpha=0.7)
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Line Plot of Random Variables X and Y')
plt.legend()
plt.show()

#%%
#5
plt.figure()
plt.hist(x, bins=30, alpha=0.7, label='Random Variable X')
plt.hist(y, bins=30, alpha=0.7, label='Random Variable Y')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Random Variables X and Y')
plt.legend()
plt.show()

#%%
#6
url = "https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/tute1.csv"
df = pd.read_csv(url)

print(df.tail(5))


#%%
#7
corr_sales_adbudget = pearson_correlation(df['Sales'], df['AdBudget'])
corr_sales_gdp = pearson_correlation(df['Sales'], df['GDP'])
corr_adbudget_gdp = pearson_correlation(df['AdBudget'], df['GDP'])

print(f"The sample Pearson's correlation coefficient between Sales & AdBudget is: {corr_sales_adbudget:.2f}")
print(f"The sample Pearson's correlation coefficient between Sales & GDP is: {corr_sales_gdp:.2f}")
print(f"The sample Pearson's correlation coefficient between AdBudget & GDP is: {corr_adbudget_gdp:.2f}")


#%%
#8
plt.figure()
plt.scatter(df['Sales'], df['AdBudget'], alpha=0.7)
plt.xlabel("Sales")
plt.ylabel("AdBudget")
plt.title(f"{"Scatter Plot Between Sales and AdBudget"} (Correlation: {corr_sales_adbudget:.2f})")
plt.grid(True)
plt.show()

#%%
#9
plt.figure()
plt.scatter(df['Sales'], df['GDP'], alpha=0.7)
plt.xlabel("Sales")
plt.ylabel("GDP")
plt.title(f"{"Scatter Plot Between Sales and GDP"} (Correlation: {corr_sales_gdp:.2f})")
plt.grid(True)
plt.show()

#%%
#10
plt.figure()
plt.scatter(df['GDP'], df['AdBudget'], alpha=0.7)
plt.xlabel("GDP")
plt.ylabel("AdBudget")
plt.title(f"{"Scatter Plot Between GDP and AdBudget"} (Correlation: {corr_adbudget_gdp:.2f})")
plt.grid(True)
plt.show()

#%%
#11
plt.figure()
plt.plot(df['Date'], df['Sales'], label='Sales', alpha=0.7)
plt.plot(df['Date'], df['AdBudget'], label='AdBudget', alpha=0.7)
plt.plot(df['Date'], df['GDP'], label='GDP', alpha=0.7)

tick_positions = df['Date'][::7]  # Every week
plt.xticks(tick_positions, rotation=30)

plt.xlabel('Date')
plt.ylabel('Value')
plt.title('Line Plot of Sales, AdBudget, and GDP')
plt.legend()
plt.show()

#%%
#12
plt.figure()
plt.hist(df['Sales'], bins=30, alpha=0.5, label='Sales')
plt.hist(df['AdBudget'], bins=30, alpha=0.5, label='AdBudget')
plt.hist(df['GDP'], bins=30, alpha=0.5, label='GDP')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Sales, AdBudget, and GDP')
plt.legend()
plt.show()