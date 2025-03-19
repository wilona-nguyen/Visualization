import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
url1="https://github.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/raw/refs/heads/main/Sample%20-%20Superstore.xls"

df = pd.read_excel(url1)
plt.figure(figsize=(10,5))

(Top_region) = df.loc[:,['Quantity','Region','Sales','Discount','Profit']].groupby('Region').sum().sort_values('Sales', ascending=False)
Top_region.reset_index(inplace=True)

plt.bar(Top_region['Region'], Top_region['Sales'], color = '#FF6F61', label = 'Sales')

plt.bar(Top_region['Region'],Top_region['Quantity'], color = 'green', edgecolor = 'Red', linewidth = 1,
        label = 'Quantity')

plt.xlabel('Region', fontsize = 15)
plt.legend()
plt.title('Region Revenue', fontsize = 18)
plt.tight_layout()
plt.grid()

for k,v in Top_region['Sales'].astype('int').items():
    plt.text(k, v - 150000, '$' + str(v), fontsize = 12,
             color = 'k',
             horizontalalignment = 'center')


plt.show()
print(Top_region)

#%% #%%
# ========================
# 3D contour plot
# =====================
def f(x1,x2):
    return (x1+x2)**3 + x1**2 - x2**2 - 12 * x1 * x2  + x1 + x2 + 1
x = np.linspace(-2,2, 50)
y = np.linspace(-2,2, 50)
X, Y = np.meshgrid(x,y)
Z = f(X,Y)
fig = plt.figure()
ax = fig.add_subplot(111, projection ='3d')
ax.plot_wireframe(X,Y,Z)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8,8))
level = np.arange(-3,5, 0.5)
CS = plt.contour(X,Y, Z, level, color = 'k')
plt.clabel(CS, inline = 1, fontsize = 10)
plt.title('Simple contour plot')
plt.axis('equal')
plt.xlim([-2,2])
plt.ylim([-2,2])
plt.grid()
plt.show()

#%% Fill Between plots
font1 = {'family':'serif', 'color':'blue', 'size':20}
font2 = {'family':'serif', 'color':'darkred', 'size':15}
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)
plt.plot(x, y1, '--',label = 'sine curve', color = 'blue',
         lw = 2)
plt.plot(x, y2,'-.', label = 'cosine curve', color = 'red',
         lw = 2)
plt.fill_between(x,y1,y2, where=(y1>y2), alpha = 0.4,color = 'c')
plt.fill_between(x,y1,y2, where=(y1<y2), alpha = 0.4,color = 'orange')
plt.annotate('area where sine is greater than cosine', xy=(2,0.25),xytext=(3,1),
             fontsize = 13,
             arrowprops=dict(facecolor = 'green', arrowstyle='->',
                             color = 'green'))
plt.title('sine', fontdict = font1)
plt.xlabel('x-axis', fontdict = font2)
plt.ylabel('y-axis', fontdict = font2)
plt.legend(prop = {'size':13})
plt.grid()
plt.tight_layout()
plt.show()

#%% #%%
# ========================
# Twin axes plot
# =====================
x = np.arange(1,6)
y1 = np.array([10,15,7, 12, 9])
y2 = np.array([200,300,150, 250, 180])

fig, ax = plt.subplots(figsize=(8,4))
ax.bar(x,y1, color = 'b', alpha = 0.7, label = 'Sales')
ax.set_xlabel('Month')
ax.set_ylabel('Sales', color = 'b')
ax.set_ylim(0,20)

# Twin axis
ax1 = ax.twinx()
ax1.plot(x,y2, color = 'red',marker = 'o',
         label = 'Revenue')

ax1.set_ylabel('Revenue', color = 'r')
ax1.set_ylim(0,400)

fig.legend(loc = 'upper left', bbox_to_anchor = (0.15,0.85))
plt.title('Sales and Revenue', fontdict=font2)
plt.grid()
plt.tight_layout()
plt.show()