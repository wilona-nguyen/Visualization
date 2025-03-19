import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/CONVENIENT_global_confirmed_cases.csv"
sns.set_style('ticks')
df = pd.read_csv(url)
print(f'{df.columns}')

# Using seaborn package compare the total number confirmed
# covid cases between US and Brazil set the linewidth 3
# figure size (11.7, 8.27)

fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
df['date'] = pd.date_range(
    start='01-23-20',
    end = '11-23-20'
)
sns.lineplot(x = 'date',
             y = 'US',
             linewidth = 3,
             label = 'US',
             data=df)

sns.lineplot(x = 'date',
             y = 'Brazil',
             linewidth = 3,
             label = 'Brazil',
             data=df)
plt.grid()
plt.tight_layout()
plt.show()
#%%
print(sns.get_dataset_names())
car = sns.load_dataset('car_crashes')
flights = sns.load_dataset('flights')
iris = sns.load_dataset('iris')
tips = sns.load_dataset('tips')
mpg = sns.load_dataset('mpg')
diamonds = sns.load_dataset('diamonds')
titanic = sns.load_dataset('titanic')
store = pd.read_excel('Sample - Superstore.xls')
#%%
sns.set(font_scale = 1.1)
sns.lineplot(data=flights,
             x = 'year',
             y = 'passengers',
             errorbar = None,
             linewidth = 3,
             color = 'green',
             hue='month')
plt.title('Flight Dataset')
plt.grid(axis='y')
plt.tight_layout()
plt.show()
#%% Plot box plot for Quantity and Discount for store datase
sns.boxplot(data=store[['Quantity','Discount']],
            # palette=sns.color_palette('Paired')
            palette='Set2',
            showfliers = True,
            showmeans= True,
            whiskerprops = dict(linestyle = 'dashed'),
            flierprops = dict(marker='o', markersize = 10,markeredgecolor = 'b'),
            medianprops = dict(linestyle = 'solid',linewidth = 2)
            )
plt.grid(axis='y')
plt.legend()
plt.tight_layout()
plt.show()

#%% Boxen plot for titanic dataset
plt.figure(figsize=(8,10))
sns.catplot(kind='boxen',
            data=titanic,
            x = 'age',
            y = 'class',
            hue='sex')

plt.tight_layout()
plt.show()
#%%
import plotly.express as px
df = px.data.gapminder()
df_canada = df.query("country=='Canada'")
df_2007 = df.query("year==2007")

#%%
df4 = mpg.groupby(['cylinders']).count()
df4.reset_index(inplace=True)
df4 = df4.sort_values(['mpg'],ascending=False)
df4['count1'] = df4['mpg']
df4['cylinders'] = df4['cylinders'].astype(str)
sns.barplot(data=df4,
                 x = 'cylinders',
                 y = 'count1',
            palette='Set2'
            )
plt.tight_layout()
plt.show()
#%% Alternative
sns.countplot(data=mpg,
              x = 'cylinders',
              palette='Set2',
              order = mpg.cylinders.value_counts().index)

plt.tight_layout()
plt.show()

#%% =============Cluster map
# Data simulation
import  numpy as np
np.random.seed(2)

data = np.random.rand(6, 6)
sns.clustermap(data=data,
               figsize=(8,8),
               cmap = 'vlag',
               vmin=-1,
               vmax=1
               )
plt.show()
#%%
g = sns.PairGrid(data=iris, hue='species')
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
g.add_legend()
plt.show()

#%%
sns.set(font_scale=1.5)
g = sns.pairplot(data=iris,
             hue='species',
             height=2.5,
             diag_kind='kde',
            )
g.add_legend()
plt.show()