#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import boxcox
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
from scipy import stats
import scipy.stats as st
from normal_test import ks_test, shapiro_test, da_k_squared_test

#%%
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/weight-height.csv'

df = pd.read_csv(url)
print(df.head())

#%% - 1
female_df = df[df['Gender'] == 'Female'].head(100)

plt.figure(figsize=(10, 6))
plt.plot(female_df.index, female_df['Height'], label='Height (inches)')
plt.plot(female_df.index, female_df['Weight'], label='Weight (pounds)')
plt.xlabel('Observation Number')
plt.ylabel('Value')
plt.title('Raw Data: Height and Weight of First 100 Female Samples')
plt.legend()
plt.grid(True)
plt.show()

#%% - 1
mean_height = female_df['Height'].mean()
var_height = female_df['Height'].var()

mean_weight = female_df['Weight'].mean()
var_weight = female_df['Weight'].var()

print(f"Sample Mean (Height): {mean_height:.2f}")
print(f"Sample Variance (Height): {var_height:.2f}")
print(f"Sample Mean (Weight): {mean_weight:.2f}")
print(f"Sample Variance (Weight): {var_weight:.2f}")


#%% - 2
plt.figure(figsize=(10, 6))
plt.hist(female_df['Height'], bins=15, alpha=0.6, label='Height (inches)')
plt.hist(female_df['Weight'], bins=15, alpha=0.6, label='Weight (pounds)')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Histogram of Height and Weight (First 100 Female Samples)')
plt.legend()
plt.grid(True)
plt.show()

#%% - 3
std_height = female_df['Height'].std()
std_weight = female_df['Weight'].std()

female_df['z_height'] = (female_df['Height'] - mean_height) / std_height
female_df['z_weight'] = (female_df['Weight'] - mean_weight) / std_weight

plt.figure(figsize=(10, 6))
plt.plot(female_df.index, female_df['z_height'], label='Z-score Height')
plt.plot(female_df.index, female_df['z_weight'], label='Z-score Weight')
plt.xlabel('Observation Number')
plt.ylabel('Z-score')
plt.title('Transformed Data: Z-scores of Height and Weight (First 100 Female Samples)')
plt.legend()
plt.grid(True)
plt.show()

#%% - 4
plt.figure(figsize=(10, 6))
plt.hist(female_df['z_height'], bins=15, alpha=0.6, label='Z-score Height')
plt.hist(female_df['z_weight'], bins=15, alpha=0.6, label='Z-score Weight')
plt.xlabel('Z-score')
plt.ylabel('Frequency')
plt.title('Histogram of Z-Transformed Height and Weight (First 100 Female Samples)')
plt.legend()
plt.grid(True)
plt.show()

#%% - 5
fig, axs = plt.subplots(2, 2, figsize=(14, 10))

# 1,1 - Question 1: Raw data line plot
axs[0, 0].plot(female_df.index, female_df['Height'], label='Height (inches)')
axs[0, 0].plot(female_df.index, female_df['Weight'], label='Weight (pounds)')
axs[0, 0].set_title('Raw Data: Height and Weight')
axs[0, 0].set_xlabel('Observation Number')
axs[0, 0].set_ylabel('Value')
axs[0, 0].legend()
axs[0, 0].grid(True)

# 1,2 - Question 3: Z-transformed line plot
axs[0, 1].plot(female_df.index, female_df['z_height'], label='Z-score Height')
axs[0, 1].plot(female_df.index, female_df['z_weight'], label='Z-score Weight')
axs[0, 1].set_title('Z-Transformed Data')
axs[0, 1].set_xlabel('Observation Number')
axs[0, 1].set_ylabel('Z-score')
axs[0, 1].legend()
axs[0, 1].grid(True)

# 2,1 - Question 2: Raw data histogram
axs[1, 0].hist(female_df['Height'], bins=15, alpha=0.6, label='Height (inches)')
axs[1, 0].hist(female_df['Weight'], bins=15, alpha=0.6, label='Weight (pounds)')
axs[1, 0].set_title('Histogram: Raw Height and Weight')
axs[1, 0].set_xlabel('Value')
axs[1, 0].set_ylabel('Frequency')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 2,2 - Question 4: Z-transformed histogram
axs[1, 1].hist(female_df['z_height'], bins=15, alpha=0.6, label='Z-score Height')
axs[1, 1].hist(female_df['z_weight'], bins=15, alpha=0.6, label='Z-score Weight')
axs[1, 1].set_title('Histogram: Z-Transformed Height and Weight')
axs[1, 1].set_xlabel('Z-score')
axs[1, 1].set_ylabel('Frequency')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()

#%% - 6
females = df[df['Gender'] == 'Female']

mean_weight = females['Weight'].mean()
std_weight = females['Weight'].std()
median_weight = females['Weight'].median()

mean_height = females['Height'].mean()
std_height = females['Height'].std()
median_height = females['Height'].median()

prob_weight_above_170 = 1 - st.norm.cdf(170, loc=mean_weight, scale=std_weight)
prob_height_above_66 = 1 - st.norm.cdf(66, loc=mean_height, scale=std_height)

print(f"Sample mean of the lady’s weight is {mean_weight:.2f} lb")
print(f"Sample mean of the lady’s height is {mean_height:.2f} inches.")
print(f"Sample std of the lady’s weight is {std_weight:.2f} lb.")
print(f"Sample std of the lady’s height is {std_height:.2f} inches.")
print(f"The median of the lady’s weight is {median_weight:.2f} lb.")
print(f"The median of the lady’s height is {median_height:.2f} inches.")
print(f"The probability that a lady weighs more than 170lb is {prob_weight_above_170*100:.2f}%")
print(f"The probability that a lady be taller than sixty-six inches is {prob_height_above_66*100:.2f}%")

#%% - 7
female_height = df[df['Gender'] == 'Female']['Height']

plt.figure(figsize=(6, 6))
qqplot(female_height, line='s')
plt.title('Q-Q Plot of Female Height')
plt.grid(True)
plt.show()

#%% - 8
female_weight = df[df['Gender'] == 'Female']['Weight']

# Create QQ plot
plt.figure(figsize=(6, 6))
qqplot(female_weight, line='s')
plt.title('Q-Q Plot of Female Weight')
plt.grid(True)
plt.show()

#%% - 9
ks_test(female_df['Weight'], 'Weight Female')
ks_test(female_df['Height'], 'Height Female')

#%% - 10
shapiro_test(female_df['Weight'], 'Weight')
shapiro_test(female_df['Height'], 'Height')

#%% - 11
da_k_squared_test(female_df['Weight'], 'Weight')
da_k_squared_test(female_df['Height'], 'Height')

#%% - 12

female_heights = df[df['Gender'] == 'Female']['Height']

Q1 = female_heights.quantile(0.25)
Q3 = female_heights.quantile(0.75)
IQR = Q3 - Q1


lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR


print(f"Q1 and Q3 of the female height is {Q1:.2f} inches & {Q3:.2f} inches.")
print(f"IQR for the female height is {IQR:.2f} inches.")
print(f"Any height < {lower_bound:.2f} inches and height > {upper_bound:.2f} inches is an outlier. \n")

#%% - 13
plt.figure(figsize=(6, 8))
plt.boxplot(female_heights, vert=True, patch_artist=True)
plt.title('Boxplot of Female Height')
plt.ylabel('Height (inches)')
plt.grid(True)
plt.show()

#%% - 14
cleaned_female_heights = female_heights[(female_heights >= lower_bound) & (female_heights <= upper_bound)]

plt.figure(figsize=(6, 8))
plt.boxplot(cleaned_female_heights, vert=True, patch_artist=True)
plt.title('Boxplot of Female Height (Outliers Removed)')
plt.ylabel('Height (inches)')
plt.grid(True)
plt.show()

#%% - 15
female_weights = df[df['Gender'] == 'Female']['Weight']

Q1 = female_weights.quantile(0.25)
Q3 = female_weights.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print(f"Q1 and Q3 of the female weight is {Q1:.2f} lb & {Q3:.2f} lb.")
print(f"IQR for the female weight is {IQR:.2f} lb.")
print(f"Any weight < {lower_bound:.2f} lb and weight > {upper_bound:.2f} lb is an outlier.")

#%% - 16
plt.figure(figsize=(6, 8))
plt.boxplot(female_weights, vert=True, patch_artist=True)
plt.title('Boxplot of Female Weight')
plt.ylabel('Weight (lb)')
plt.grid(True)
plt.show()

#%% - 17
cleaned_weights = female_weights[(female_weights >= lower_bound) & (female_weights <= upper_bound)]

plt.figure(figsize=(6, 8))
plt.boxplot(cleaned_weights, vert=True, patch_artist=True)
plt.title('Boxplot of Female Weight (Outliers Removed)')
plt.ylabel('Weight (lb)')
plt.grid(True)
plt.show()







