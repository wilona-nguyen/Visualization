#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from prettytable import PrettyTable

#%%
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/mnist_test.csv'

df = pd.read_csv(url)

#%% - 1a

plt.figure(figsize=(12,12))
k =  0
for i in range(10):
    for j in range(10):
        k += 1
        plt.subplot(10,10, k)
        pic = df.iloc[k-1,1:].values.reshape(28,28)
        plt.imshow(pic)

plt.tight_layout()
plt.show()

#%% - 1b
# zero = df[df['label'] == 0].head(10)
#
# plt.figure(figsize=(12,12))
# k =  0
# for i in range(10):
#     for j in range(10):
#         k += 1
#         plt.subplot(1,10, k)
#         pic = zero.iloc[k-1,1:].values.reshape(28,28)
#         plt.imshow(pic)
#
# plt.tight_layout()
# plt.show()

#%% - 2
df = sns.load_dataset('diamonds')
print(df.isna().sum())

#%% - 3


table = PrettyTable(['Number', 'Cut Name'])

for i in range(len(df['cut'].unique())):
    table.add_row([i + 1, df['cut'].unique()[i]])

print(table.get_string(title="Diamond Dataset - Various cuts"))

#%% - 4
table_color = PrettyTable(['Number', 'Color Name'])

for i in range(len(df['color'].unique())):
    table_color.add_row([i + 1, df['color'].unique()[i]])

print(table_color.get_string(title="Diamond Dataset - Color Name"))

#%% - 5
table_clarity = PrettyTable(['Number', 'Clarity Name'])

for i in range(len(df['clarity'].unique())):
    table_color.add_row([i + 1, df['clarity'].unique()[i]])

print(table_color.get_string(title="Diamond Dataset - Clarity Name"))

#%% - 6
df_sale_cut = pd.DataFrame({'cut': df['cut'].value_counts().index.tolist(),
                            'count': df['cut'].value_counts().values.tolist()})

plt.barh(df_sale_cut['cut'], df_sale_cut['count'])

plt.xlabel("Number of Sales")
plt.ylabel("Cut")
plt.title('Sales count per cut')
plt.grid()
plt.tight_layout()
plt.show()

cut_max_sale = df['cut'].value_counts().idxmax()
print(f"The diamond with {cut_max_sale} cut has the maximum number of sales.")

cut_min_sale = df['cut'].value_counts().idxmin()
print(f"The diamond with {cut_min_sale} cut has the minimum number of sales.")

#%% - 7

df_sale_color = pd.DataFrame({'color': df['color'].value_counts().index.tolist(),
                            'count': df['color'].value_counts().values.tolist()})

plt.barh(df_sale_color['color'], df_sale_color['count'])

plt.xlabel("Number of Sales")
plt.ylabel("Color")
plt.title('Sales count per color')
plt.grid()
plt.tight_layout()
plt.show()

color_max_sale = df['color'].value_counts().idxmax()
print(f"The diamond with {color_max_sale} color has the maximum number of sales.")

color_min_sale = df['color'].value_counts().idxmin()
print(f"The diamond with {color_min_sale} color has the minimum number of sales.")

#%% - 8
df_sale_clarity = pd.DataFrame({'clarity': df['clarity'].value_counts().index.tolist(),
                            'count': df['clarity'].value_counts().values.tolist()})

plt.barh(df_sale_clarity['clarity'], df_sale_clarity['count'])

plt.xlabel("Number of Sales")
plt.ylabel("Clarity")
plt.title('Sales count per clarity')
plt.tight_layout()
plt.grid()
plt.show()


clarity_max_sale = df['clarity'].value_counts().idxmax()
print(f"The diamond with {clarity_max_sale} clarity has the maximum number of sales.")

clarity_min_sale = df['clarity'].value_counts().idxmin()
print(f"The diamond with {clarity_min_sale} clarity has the minimum number of sales.")

#%% - 9
plt.figure()

fig, ax = plt.subplots(1,3)

ax[0].barh(df_sale_cut['cut'], df_sale_cut['count'])
ax[1].barh(df_sale_color['color'], df_sale_color['count'])
ax[2].barh(df_sale_clarity['clarity'], df_sale_clarity['count'])

plt.show()

#%% - 10
df_sale_cut['Sale_Percentage'] = round(df_sale_cut['count'] / sum(df_sale_cut['count']) * 100, 2)
df_sale_cut = df_sale_cut.set_index('cut')

#%%
explode = []

for cat in df_sale_cut.index:
   explode.append([0.2 if cat == df_sale_cut['Sale_Percentage'].idxmin() else 0.03 for cat in df_sale_cut.index])

plt.figure()

plt.pie(df_sale_cut['Sale_Percentage'], labels=df_sale_cut.index, autopct='%1.2f%%', explode = explode[1])

plt.title('Sales count per cut in %')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

print(f"The diamond with {df_sale_cut['Sale_Percentage'].idxmax()} cut has the maximum number of sales with {df_sale_cut['Sale_Percentage'].max()} % sales count")

print(f"The diamond with {df_sale_cut['Sale_Percentage'].idxmin()} cut has the maximum number of sales with {df_sale_cut['Sale_Percentage'].min()} % sales count")

#%% - 11
df_sale_color['Sale_Percentage'] = round(df_sale_color['count'] / sum(df_sale_color['count']) * 100, 2)
df_sale_color = df_sale_color.set_index('color')

#%%
explode = []

for cat in df_sale_color.index:
   explode.append([0.2 if cat == df_sale_color['Sale_Percentage'].idxmin() else 0.03 for cat in df_sale_color.index])

plt.figure()
plt.pie(df_sale_color['Sale_Percentage'], labels=df_sale_color.index, autopct='%1.2f%%', explode = explode[1])

plt.title('Sales count per color in %')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

#%%
print(f"The diamond with {df_sale_color['Sale_Percentage'].idxmax()} cut has the maximum number of sales with {df_sale_color['Sale_Percentage'].max()} % sales count")

print(f"The diamond with {df_sale_color['Sale_Percentage'].idxmin()} cut has the maximum number of sales with {df_sale_color['Sale_Percentage'].min()} % sales count")


#%% -12

df_sale_clarity['Sale_Percentage'] = round(df_sale_clarity['count'] / sum(df_sale_clarity['count']) * 100, 2)
df_sale_clarity = df_sale_clarity.set_index('clarity')

#%%
explode = []

for cat in df_sale_clarity.index:
   explode.append([0.2 if cat == df_sale_clarity['Sale_Percentage'].idxmin() else 0.03 for cat in df_sale_clarity.index])

plt.figure()
plt.pie(df_sale_clarity['Sale_Percentage'], labels=df_sale_clarity.index, autopct='%1.2f%%', explode = explode[1])

plt.title('Sales count per clarity in %')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()

#%%
print(f"The diamond with {df_sale_clarity['Sale_Percentage'].idxmax()} cut has the maximum number of sales with {df_sale_clarity['Sale_Percentage'].max()} % sales count")

print(f"The diamond with {df_sale_clarity['Sale_Percentage'].idxmin()} cut has the maximum number of sales with {df_sale_clarity['Sale_Percentage'].min()} % sales count")

#%% - 13
vs1_df = df[df['clarity'] == 'VS1']
cut_list = df['cut'].unique().tolist()
color_list = df['color'].unique().tolist()

dd = vs1_df.groupby(by=['cut', 'color']).agg({'price': 'mean'})
# dd = pd.DataFrame({'cut': cut_list, 'color': color_list, 'price': dd['price'].values})

#%%
list = []
for i in range(len(dd.index)):
    cut_color_price = dd.iloc[i]['price']
    list.append(cut_color_price)
print(list)
#%%
vs1_table = PrettyTable(['D', 'E', 'F', 'G', 'H', 'I', 'J', 'Max', 'Min'])

# for cut_type in cut_list:
#     row = [cut_type]
#     # print(row)
#     for i in range(len(list)):
#         value = list[i]
#         row.append(round(value, 2))
#     vs1_table.add_row(row)
#     print(row)

print(vs1_table)
