#%%
import seaborn as sns
import pandas as pd
import dash
import numpy as np
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import matplotlib.pyplot as plt


#%%

tips = sns.load_dataset('tips')

#%% - pie plot that show relationship between 'day' and 'total_bill'

labels = ['Thur', 'Fri', 'Sat', 'Sun']




#%% - count sum for total_bill by each 'day'
day_totbil_df = tips[['day', 'total_bill']]
#%%
tot_bill = day_totbil_df.groupby('day').sum()

#%%
plt.pie(tot_bill['total_bill'], labels = labels, startangle = 90, autopct='%1.1f%%')
plt.title('Total Bill')
plt.legend(loc = 'upper right')
plt.show()

#%%
sns.displot(x = 'day', y = 'total_bill', data = tips, kind = 'pie')
plt.show()

#%%
tips_group_time = tips.loc[:,['day', 'tip']].groupby(['day']).sum()
explode = (.01, .3, .01, .01)

tips_group_time.plot(kind = 'pie',
                     y = 'tip',
                     autopct = '%1.0f%%',
                     startangle = 60,
                     grid = True,
                     fontsize = 15,)

plt.tight_layout()
plt.show()

#%%
tips.loc[:,['day', 'tip']].groupby(['day']).sum().plot(kind = 'pie',
                     y = 'tip',
                     autopct = '%1.0f%%',
                     startangle = 60,
                     grid = True,
                     fontsize = 15,)

plt.tight_layout()
plt.show()

#%%
drop_a = ['day', 'time', 'sex']
drop_b = ['tip', 'total_bill']

fro
    fig = px.pie(tips, values=drop_b, names=drop_a)
fig.show()