import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


#%%
url1 = 'https://github.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/raw/refs/heads/main/Sample%20-%20Superstore.xls'
#url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/Sample%20-%20Superstore.xls'
df = pd.read_excel(url1)
print(df.head())

#%%
plt.figure(figsize = (10,5))

(Top_region) = df.loc[:, ['Quantity', 'Region', 'Sales', 'Discount', 'Profit']].groupby('Region').sum().sort_values(by='Sales', ascending=False).head(10)
Top_region.reset_index(inplace=True)
plt.bar(Top_region['Region'], Top_region['Sales'], color = '#FF6F61', label = 'Sales')
plt.bar(Top_region['Region'], Top_region['Quantity'], color = 'green', edgecolor = 'black', linewidth = 1, label = 'Quantity')
plt.xlabel('Region', fontsize = 15)
plt.title('Region Revenue', fontsize = 18)
plt.grid()
plt.legend()
plt.tight_layout()


for k,v in Top_region['Sales'].astype(int).items():
    plt.text(k, v - 150000,  '$' + str(v), color = 'black', fontsize = 12, horizontalalignment = 'center')

plt.show()

#%% - 3D Contour plot
def f(x1, x2):
    return (x1 + x2)**4 - 12 * x1 * x2 + x1 + x2 + 1

x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)

Z = f(X,Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.plot_wireframe(X, Y, Z)

plt.tight_layout()
plt.show()

#%%
plt.figure(figsize=(8,8))
level = np.arange(-3, 5, 0.5)
CS = plt.contour(X, Y, Z, level)
plt.clabel(CS, inline = 1, fontsize = 10)
plt.title('Simple Contour Plot', fontsize = 18)
plt.axis('equal')
plt.xlim([-1, 1])
plt.ylim([-1, 1])
plt.grid()
plt.show()

#%% - Fill Between plots
font1 = {'family' : 'serif', 'color': 'blue', 'size':20}
font2 = {'family' : 'serif', 'color': 'darkred', 'size':15}

x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, '--', label = 'sine curve', color = 'blue', lw = 2)
plt.plot(x, y2, '-.', label = 'cosine curve', color = 'red', lw = 2)
plt.fill_between(x, y1, y2, where = y1 > y2, facecolor = 'c', alpha = 0.5)
plt.fill_between(x, y1, y2, where = y1 <= y2, facecolor = 'orange', alpha = 0.5)
plt.annotate('area where sine is greater than cosine', xy = (2,0.25),
             xytext = (2,0.5),
             fontsize = 13,
             arrowprops = dict(facecolor = 'green',
                               arrowstyle = '->',
                               color = 'green'))
plt.title('sine', fontdict = font1)
plt.xlabel('x-axis', fontdict = font2)
plt.ylabel('y-axis', fontdict = font2)
plt.legend(prop = {'size': 15})
plt.grid()
plt.tight_layout()
plt.show()

#%% - Twin axes plot
x = np.arange(1, 6)
y1 = np.array([10, 15, 7, 12, 9])
y2 = np.array([200, 300, 150, 250, 180])

fig, ax = plt.subplots(figsize = (8,4))

ax.bar(x, y1, color = 'blue', alpha = 0.7, label = 'Sales')
ax.set_xlabel('Month')
ax.set_ylabel('Sales', color = 'g')
ax.set_ylim([0, 20])

ax1 = ax.twinx()
ax1.plot(x, y2, color = 'red', marker = 'o', label = "Revenue")
ax1.set_ylabel('Revenue', color = 'r')
ax1.set_ylim(0, 400)

fig.legend(loc = 'upper left', bbox_to_anchor = (0.1, 0.90))
plt.title('Sales and Revenue', fontsize = 18)
plt.grid()
plt.tight_layout()
plt.show()



