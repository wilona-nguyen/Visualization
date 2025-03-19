import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from prettytable import PrettyTable
#%%
#Q1
url = 'https://github.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/raw/refs/heads/main/Sample%20-%20Superstore.xls'

df = pd.read_excel(url)
col_to_remove = ['Row ID', 'Order ID', 'Customer ID','Customer Name', 'Postal Code',
                 'Product ID','Order Date', 'Ship Date', 'Country', 'Segment']
df = df.drop(col_to_remove, axis=1)
print(df.head().to_string())

#%%

#Q2
agg_df = df.groupby('Category').agg({
    'Profit': 'sum',
    'Quantity': 'sum',
    'Discount': 'sum',
    'Sales': 'sum'
}).reset_index()


agg_df[['Profit', 'Discount', 'Sales']] = agg_df[['Profit', 'Discount', 'Sales']].round(2)


columns = ['Profit', 'Quantity', 'Discount', 'Sales']
percentages = agg_df[columns].div(agg_df[columns].sum(axis=0), axis=1) * 100


results = {}
for col in columns:
    max_cat = agg_df['Category'][percentages[col].idxmax()]
    min_cat = agg_df['Category'][percentages[col].idxmin()]
    results[col] = (max_cat, min_cat)

for k, v in results.items():
    print(f"{k}: Max: {v[0]}, Min: {v[1]}")

explode = []
for col in columns:
    explode.append([0.1 if cat == results[col][1] else 0.0 for cat in agg_df['Category']])

fig, axes = plt.subplots(2, 2, figsize=(18, 18))
axes = axes.flatten()

for i, col in enumerate(columns):
    axes[i].pie(percentages[col], labels=agg_df['Category'], explode=explode[i], autopct='%1.2f%%',
                textprops={'fontsize': 30})
    axes[i].set_title(col, fontdict={'family': 'serif', 'color': 'blue', 'size': 35})

plt.tight_layout()
plt.show()

agg_df[columns] = agg_df[columns].map(lambda x: f"{x:.2f}")


#%%
#Q3

columns = ['Profit', 'Quantity', 'Discount', 'Sales']
agg_df[columns] = agg_df[columns].apply(pd.to_numeric)

x = PrettyTable()
x.field_names = ['', 'Sales ($)', 'Quantity', 'Discounts ($)', 'Profit ($)']

x.add_rows(agg_df.values.tolist())

max_values = []
min_values = []
max_features = []
min_features = []

for col in columns:
    max_values.append(agg_df[col].max())
    min_values.append(agg_df[col].min())
    max_features.append(agg_df[col].idxmax())
    min_features.append(agg_df[col].idxmin())

x.add_row(["Maximum Value", *max_values])
x.add_row(["Minimum Value", *min_values])

x.add_row(["Maximum feature"] + [agg_df['Category'].iloc[i] for i in max_features])
x.add_row(["Minimum feature"] + [agg_df['Category'].iloc[i] for i in min_features])

x.float_format = '.2'

print(x.get_string(title="Super store - Category"))



#%%
#Q4
grouped_data = df.groupby('Sub-Category').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()
grouped_data = grouped_data.sort_values(by='Sales', ascending=False)
top_10_data = grouped_data.head(10)

fig, ax1 = plt.subplots(figsize=(20, 8))

ax1.set_axisbelow(True)
ax1.grid(True, which='both', axis='both', linewidth=0.5, zorder=0)

bars = ax1.bar(top_10_data['Sub-Category'], top_10_data['Sales'], label = 'Sales',
               color='#95DEE3', edgecolor='blue', width=0.4)
ax1.set_xlabel('Sales', fontsize=25)
ax1.set_ylabel('USD ($)', fontsize=25)
ax1.set_ylim([-50000, 350000])




for i, bar in enumerate(bars):
    height = bar.get_height()
    if i < 2:
        va = 'center'
        y_offset = -30000
    else:
        va = 'bottom'
        y_offset = 5000

    ax1.text(bar.get_x() + bar.get_width() / 2., height + y_offset,
             f'${height:,.2f}', ha='center', va=va,
             fontsize=15, color='black', rotation=90, zorder=3)



ax2 = ax1.twinx()
ax2.plot(top_10_data['Sub-Category'], top_10_data['Profit'],
         color='red', linewidth=4, marker='o', zorder=4,
         label = 'Profit')

ax2.set_ylabel('USD ($)', fontsize=25)
ax2.set_ylim([-50000, 350000])



ax1.tick_params(axis='x', labelsize=20)
ax1.tick_params(axis='y', labelsize=20)
ax2.tick_params(axis='y', labelsize=20)

ax2.grid(False)



plt.title('Sales and Profit per Sub-Category', fontsize=30)

handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper right', bbox_to_anchor=(1.1, 1.1))

plt.xticks(rotation=45)
plt.show()


#%%
#Q5
font1 = {'family' : 'serif', 'color': 'blue', 'size':20}
font2 = {'family' : 'serif', 'color': 'darkred', 'size':15}

x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)


plt.plot(x, y1, '--', label = 'sine curve', color = 'blue', lw = 3)
plt.plot(x, y2, '-.', label = 'cosine curve', color = 'red', lw = 3)
plt.fill_between(x, y1, y2, where = y1 > y2, facecolor = 'green', alpha = 0.3)
plt.fill_between(x, y1, y2, where = y1 <= y2, facecolor = 'orange', alpha = 0.3)
plt.annotate('area where sine is greater than cosine', xy = (2,0.25),
             xytext = (3, 1),
             fontsize = 12,
             arrowprops = dict(facecolor = 'green',
                               arrowstyle = '->')
             )
plt.title('Fill between x-axis and plot line', fontdict = font1)
plt.xlabel('x-axis', fontdict = font2)
plt.ylabel('y-axis', fontdict = font2)

plt.legend(fontsize = 15, loc = 'lower left')
plt.grid()
plt.tight_layout()
plt.show()



#%%
#Q6


x = np.arange(-4, 4, 0.01)
y = np.arange(-4, 4, 0.01)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))


fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')


surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', alpha = 0.7)

ax.contour(X, Y, Z, zdir='z', offset=-6, cmap='coolwarm', linewidths=1)
ax.contour(X, Y, Z, zdir='x', offset=-4, cmap='coolwarm', linewidths=1)
ax.contour(X, Y, Z, zdir='y', offset=4, cmap='coolwarm', linewidths=1)

ax.set_zlim(-6, 2)

ax.set_xlabel("X Label", fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
ax.set_ylabel("Y Label", fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})
ax.set_zlabel("Z Label", fontdict={'family': 'serif', 'color': 'darkred', 'size': 15})

ax.set_title(r"surface plot of $z = \sin \sqrt{x^2 + y^2}$",
             fontdict={'family': 'serif', 'color': 'blue', 'size': 25})

plt.show()

#%%
#Q7

# top_10_data

fig = plt.figure(figsize=(9, 7))
gs = fig.add_gridspec(2, 2)

# Bar plot across top row (gs[0, :])
ax0 = fig.add_subplot(gs[0, :])
bar_width = 0.4
x = np.arange(len(top_10_data))

ax0.set_axisbelow(True)
ax0.grid(True, which='both', axis='both', linewidth=0.5, zorder=0)

ax0.bar(x - bar_width/2, top_10_data['Sales'], width=bar_width, color='#95DEE3', edgecolor='blue', label="Sales", zorder=3)
ax0.bar(x + bar_width/2, top_10_data['Profit'], width=bar_width, color='lightcoral', edgecolor='red', label="Profit", zorder=3)

for spine in ax0.spines.values():
    spine.set_color('gray')

ax0.set_xticks(range(len(top_10_data)))
ax0.set_xticklabels(top_10_data['Sub-Category'])
ax0.set_ylabel("USD ($)", fontsize=10)
ax0.set_xlabel("Sub-Category", fontsize=10)
ax0.set_ylim([-50000, 350000])
ax0.set_title("Sales and Profit per Sub-Category", fontsize=15)
ax0.legend(fontsize=10)
ax0.tick_params(axis='both', labelsize=10)




# Pie chart data (Sales and Profit %)
columns = ['Sales', 'Profit']
percentages = agg_df[columns].div(agg_df[columns].sum(axis=0), axis=1) * 100

# Sales pie chart (gs[1, 0])
ax1 = fig.add_subplot(gs[1, 0])
ax1.pie(percentages['Sales'],
        labels=agg_df['Category'],
        autopct='%1.2f%%',
        textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax1.set_title("Sales", fontsize=15)

# Profit pie chart (gs[1, 1])
ax2 = fig.add_subplot(gs[1, 1])
ax2.pie(percentages['Profit'],
        labels=agg_df['Category'],
        autopct='%1.2f%%',
        textprops={'fontsize': 12},
        wedgeprops={'edgecolor': 'white', 'linewidth': 2})
ax2.set_title("Profit", fontsize=15)




plt.tight_layout()
plt.show()
