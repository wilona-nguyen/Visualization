import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
tips = sns.load_dataset('tips')
diamond = sns.load_dataset('diamonds')
titanic = sns.load_dataset('titanic')

np.random.seed(6401)

#%%
df = pd.DataFrame(data = np.random.rand(10,4),
                  columns = list('ABCD'))

print(df.head(3))

plt.figure()

ax = df.plot(subplots = True, #if no argument, then everything is plot together
             layout = (4,1), #row, col
             grid = True,
             figsize = (8,8),
             fontsize = 15,
             lw = 3,            #line thickness
             xlim = (0, 8),
             ylim = (0, 2),
             #title = 'dummy dataset',
             style = ['b*-', 'ro-', 'y^-', 'cD-'])

ax[0,0].set_ylabel('Mag1', fontsize = 14)
ax[1,0].set_ylabel('Mag2', fontsize = 14)
ax[2,0].set_ylabel('Mag2', fontsize = 14)
ax[3,0].set_ylabel('Mag3', fontsize = 14)
ax[3,0].set_xlabel('samples', fontsize = 14)

ax[0,0].set_title('Dummy Dataset', fontsize = 15)

plt.tight_layout()
plt.show()

#%%
import pandas as pd
data = {'Gender': ['Female', 'Male'],
        'Age': [25, 18],
        'Weight': [150, 180],
        'Location': ['CA', 'DC'],
        'Arrest Record': ['No', 'Yes']}

df = pd.DataFrame(data)
print(df)

#%%
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/store.csv'
df = pd.read_csv(url)
print(df.tail(3).to_string())

# slice to first 6000 rows with these columns ['MONTH', 'STORECODE', 'QTY', 'VALUE', 'MBRD']
#df_6000 = df.iloc[:6000,:]
df_6000 = df.loc[:6000, ['MONTH', 'STORECODE', 'QTY', 'VALUE', 'MBRD']]
print(df_6000.head())

# aggregate the df_6000 by the 'MONTH' and show the total items sold
df_month_qty = df.loc[:6000, ['MONTH', 'QTY']]
df_month_qty_agg = df_month_qty.groupby('MONTH').sum()
print(df_month_qty_agg)

#%%
df_month_qty_agg_sorted = df_month_qty.groupby('MONTH').sum().sort_values(by='QTY',
                                                                          ascending=False)

df_month_qty_agg_sorted.plot(kind = 'barh',
                      ylabel = 'Month',
                      xlabel = 'Total Items Sold',
                      grid = True,
                      fontsize = 10,
                      )
plt.tight_layout()
plt.show()

#%%
#load the tips dataset
#gender_tip = tips.loc[:,['sex','tip']]
#tips_agg_sorted = gender_tip.groupby('sex').sum()

tips_agg_sorted=tips.loc[:,['sex','tip']].groupby(['sex']).sum().sort_values(by='tip', ascending=True)
tips_agg_sorted.reset_index(inplace=True)

tips_agg_sorted.plot(kind = 'bar',
                     x = 'sex',
                     y = 'tip',
                     xlabel = 'Sex',
                     ylabel = 'USD ($)',
                     title = 'bar plot',
                     grid = True,
                     fontsize = 10)

plt.tight_layout()
plt.show()

#%%
#hist
tips[['tip', 'total_bill']].plot(kind = 'hist',
                                 bins = 50,
                                 grid = True,
                                 title = 'hist plot',
                                 alpha = 0.8,
                                 orientation = 'vertical',
                                 stacked = True,)
plt.tight_layout()
plt.show()

#%%
#box plot
tips[['total_bill', 'tip', 'sex']].plot(kind = 'box',
                                        grid = True,
                                        title = 'box plot',
                                        by = ['sex'],
                                        figsize = (8,6),
                                        fontsize = 15,)

plt.tight_layout()
plt.show()

#%%
#area plot

df = pd.DataFrame(data = np.random.rand(10,4),
                  columns = list('ABCD'))

df.plot(kind = 'area',
        grid = True,
        title = 'area plot',
        xlabel = 'x-axis',
        ylabel = 'y-axis',
        fontsize = 15,)

plt.tight_layout()
plt.show()

#%%
#scatter plot

tips.plot(kind = 'scatter',
          x = 'total_bill',
          y = 'tip',
          marker = 'o',
          color = 'red',
          s = 100,
          grid = True,
          title = 'scatter plot',
          xlabel = 'total_bill',
          ylabel = 'tip',
          fontsize = 20,)

plt.tight_layout()
plt.show()

#%%
#pie chart
tips_group_time = tips.loc[:,['day', 'tip']].groupby(['day']).sum()
explode = (.01, .3, .01, .01)

tips_group_time.plot(kind = 'pie',
                     y = 'tip',
                     autopct = '%1.0f%%',
                     explode = explode,
                     startangle = 60,
                     grid = True,
                     title = 'pie chart',
                     fontsize = 15,)

plt.tight_layout()
plt.show()

#%%
#create a new column that encode genders: Male = 0, Female = 1
# tips['gender_encoded'] = tips['sex'].map({'Male': 0, 'Female': 1})
tips['gender_encoded'] = tips['sex'].apply(lambda x: 1 if x=='Female' else 0)
#tips['Tip_percentage'] = 100*tips['tip'] / tips['total_bill']

def per(x, y):
    return 100*x/y

tips['Tip_percentage'] = tips.apply(lambda x: per(x['tip'], x['total_bill']), axis=1)

tips['Tip_status'] = tips['Tip_percentage'].apply(lambda x: 'High' if x>=20 else ('Normal' if x>=10 else 'Low'))
#tips['Tip_status'] = tips['Tip_percentage'].apply(lambda x: 'Low' if x < 10 else 'Normal' if x < 20 else 'High')


print(tips.head().to_string())
