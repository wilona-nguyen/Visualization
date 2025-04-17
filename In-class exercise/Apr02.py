#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import boxcox
from statsmodels.graphics.gofplots import qqplot
import seaborn as sns
from scipy import stats
import sys
sys.path.append("jetbrains://pycharm/navigate/reference?project=Lab%20%234.py&path")
from normal_test_func import ks_test, shapiro_test, da_k_squared_test

#%%
data = np.random.exponential(size=1000)

plt.hist(data, bins=100)
plt.show()

plt.figure()
qqplot(data, line='s')
plt.grid()
plt.show()

#%% - statistical test for normality
da_k_squared_test(data, "Raw")
ks_test(data, "Raw")
shapiro_test(data, "Raw")

sns.distplot(data, kde=True)
plt.show()

#%%
transformed_data, best_lambda = boxcox(data)
print(f"The best lambda is: {best_lambda}")

da_k_squared_test(transformed_data, "Raw")
ks_test(transformed_data, "Raw")
shapiro_test(transformed_data, "Raw")

sns.distplot(transformed_data, kde=True)
plt.show()

fig, ax = plt.subplots(figsize=(8, 4))
prob = stats.boxcox_normplot(data, -5, 5, plot=ax)
ax.axvline(best_lambda, linestyle = '--', color='r')