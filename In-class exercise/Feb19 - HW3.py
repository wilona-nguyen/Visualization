#%%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#%%
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/CONVENIENT_global_confirmed_cases.csv'
sns.set_style('ticks')

df = pd.read_csv(url)
print(f'{df.columns}')

#%%
#Using seaborn package compare the total number confirmed covid cases between US and Brazil

df['date'] = pd.date_range(start = '01-23-20',
                           end = '11-23-20')

fig, ax = plt.subplots()
fig.set_size_inches((11.7, 8.27))

sns.lineplot(x = 'date',
             y = 'US',
             linewidth = 3,
             label = 'US',
             data = df)

sns.lineplot(x = 'date',
             y = 'Brazil',
             linewidth = 3,
             label = 'Brazil',
             data = df)

plt.grid()
plt.tight_layout()
plt.show()

#%%
# print(sns.get_dataset_names())
car = sns.load_dataset('car_crashes')
flight = sns.load_dataset('flights')
iris = sns.load_dataset('iris')
tips = sns.load_dataset('tips')
mpg = sns.load_dataset('mpg')
titanic = sns.load_dataset('titanic')
brain = sns.load_dataset('brain_networks')

#%%
sns.set(font_scale =  1.1)
sns.lineplot(data=flight, x='year', y='passengers',
             errorbar=None,
             linewidth=3,
             color='green',
             hue = 'month')
plt.title('flight dataset')
plt.grid(axis='y')
plt.tight_layout()
plt.show()

#%% - Plot boxplot for quantity and discount for stor dataset
url = 'https://github.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/raw/refs/heads/main/Sample%20-%20Superstore.xls'
store = pd.read_excel(url)

#%%
sns.boxplot(data=store[['Quantity', 'Discount']],
            palette = 'Set2',
            showfliers = True,
            showmeans = True,
            whiskerprops = dict(linestyle = 'dashed'),
            flierprops = dict(marker = 'o',
                              markersize = 10,
                              markeredgecolor = 'blue'),
            medianprops = dict(linestyle = 'solid',
                               linewidth = 2),)

plt.grid(axis='y')
plt.legend()
plt.tight_layout()
plt.show()

#%% - Boxen plot for titanic dataset
sns.set_style('white')
plt.figure(figsize=(8,10))

sns.catplot(kind = 'boxen',
            data = titanic,
            x = 'age',
            y = 'class',
            hue = 'sex')

plt.tight_layout()
plt.show()

#%%
import plotly.express as px

df = px.data.gapminder()
df_canada = df.query('country == "Canada"')
df_2007 = df.query('year == 2007')

#%%
#groupby way

df_gb = mpg.groupby(['cylinders']).count()

#%%

sns.countplot(data =  mpg,
              x = 'cylinders',
              palette = 'Set2',
              order = mpg.cylinders.value_counts().index)

plt.show()

#%%
import numpy as np

np.random.seed (2)

data = np.random.rand(6,6)
sns.clustermap(data = data,
               figsize = (8,8),
               cmap = 'vlag',
               vmin = -1,
               vmax = 1)

plt.show

#%% - PairGrid
g = sns.PairGrid(data=iris, hue='species')
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()
plt.show()

#%%


