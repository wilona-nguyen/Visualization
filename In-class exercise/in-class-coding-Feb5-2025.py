#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

url = "https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/mnist_test.csv"
df = pd.read_csv(url)
print(df.shape)

pic = df[0:1].values.reshape(785)[1:].reshape(28,28)
plt.figure()
plt.imshow(pic)
plt.show()

#%% Datafarme slicing conditional selection
zero = df[df['label']==0]
pic = zero[0:1].values.reshape(785)[1:].reshape(28,28)
plt.figure()
plt.imshow(pic)
plt.show()
#%% Display the first 25 observations of the zero
# dataset in the subplot format of 5x5
plt.figure(figsize=(12,12))
k =  0
for i in range(5):
    for j in range(5):
        k += 1
        plt.subplot(5,5, k)
        pic = zero.iloc[k-1,1:].values.reshape(28,28)
        plt.imshow(pic)

plt.tight_layout()
plt.show()
#%% Check if the dataset is balanced by plotting the # of observations
# in each class using a barplot

df_agg = df.groupby(['label']).count().sort_values(by = '1x1', ascending=False)
df_agg.reset_index(inplace=True)
print(df_agg)
ax = df_agg.iloc[:,1:2].plot(kind = 'bar')
ax.set_xticklabels(df_agg['label'].values)
plt.tight_layout()
plt.show()

#%% Load left wing politician
from sklearn.datasets import fetch_lfw_people
faces = fetch_lfw_people(min_faces_per_person=20)
print(f'Image shape {faces.images.shape}')
print(f'Number of features {faces.data.shape[1]}')
print(f'Number of classes {faces.target_names}')

#%% Display the image of politician number 12
# row=62 columns = 47
X = faces.data
X1 = pd.DataFrame(X)
y = faces.target
y1 = pd.DataFrame(y, columns=['label'])
df = pd.concat([X1,y1], axis=1)

row_12 = df[df['label']==12]
pic = row_12.iloc[0,:-1].values.reshape(62,47)
plt.figure(figsize=(8,10))
plt.imshow(pic)
plt.show()

#%% Display the first 20 politicians using subplot
# 5 x 4
plt.figure(figsize=(8,10))
for i in range(20):
    pic = df.iloc[i,:-1].values.reshape(62,47)
    plt.subplot(5,4,i+1)
    plt.imshow(pic)
plt.tight_layout()
plt.show()

#%% Plot mathematical functions using plt.subplots
import numpy as np
lw = 3
font1 = {'family':'serif','color':'blue','size':15}
font2 = {'family':'serif','color':'darkred','size':15}
x = np.linspace(-10,10, 100)
x = np.array(x,dtype=np.complex64)
y1 = x ** 2
y2 = x ** (1/2)
y3 = x **3
y4 = x **(1/3)
#====
# x^2
#===

fig,ax = plt.subplots(2,2, figsize = (10,10))
ax[0,0].plot(x,y1,label = '$f(x) = x^2 $',lw = lw)
ax[0,0].legend(loc = 'center',fontsize=15,title_fontsize=15)
ax[0,0].set_title('$f(x) = x^2 $',fontdict=font1)
ax[0,0].set_xlabel('Samples', fontdict=font2)
ax[0,0].set_ylabel('Mag.',fontdict=font2)
ax[0,0].grid()

#====
# x^{1/2}
#===

ax[0,1].plot(x,y2,label = '$f(x) = \sqrt{x} $',lw = lw)
ax[0,1].legend(loc = 'center',fontsize=15,title_fontsize=15)
ax[0,1].set_title('$f(x) = \sqrt{x} $ $',fontdict=font1)
ax[0,1].set_xlabel('Samples',fontdict=font2)
ax[0,1].set_ylabel('Mag.',fontdict=font2)
ax[0,1].grid()
#====
# x^3
#===

ax[1,0].plot(x,y3,label = '$f(x) = x^3 $',lw = lw)
ax[1,0].legend(loc = 'center',fontsize=15,title_fontsize=15)
ax[1,0].set_title('$f(x) = x^3 $',fontdict=font1)
ax[1,0].set_xlabel('Samples',fontdict=font2)
ax[1,0].set_ylabel('Mag.',fontdict=font2)
ax[1,0].grid()

#====
# x^{1/3}
#===
ax[1,1].plot(x,y4,label = '$f(x) = \sqrt[3]{x} $',lw = lw)
ax[1,1].legend(loc = 'center',fontsize=15,title_fontsize=15)
ax[1,1].set_title('$f(x) = \sqrt[3]{x} $',fontdict=font1)
ax[1,1].set_xlabel('Samples',fontdict=font2)
ax[1,1].set_ylabel('Mag.',fontdict=font2)
ax[1,1].grid()

fig.tight_layout()
plt.show()

#%% Plot mathematical functions using plt.fig_add_subplot
x = np.linspace(1,10)
y = [10 ** el for el in x]
z = [2 ** el for el in x ]

fig = plt.figure(figsize=(10,8))
ax1 = fig.add_subplot(2,2,1)
ax1.plot(x,y,color = 'blue', lw = lw)
ax1.set_yscale('log')
ax1.set_font = 20
ax1.set_title(r'Log plot of $10^{x}$')
plt.grid(visible=True,which='both',axis='both')


ax2 = fig.add_subplot(2,2,2)
ax2.plot(x,y,color = 'red', lw = lw)
ax2.set_yscale('linear')
ax2.set_font = 20
ax2.set_title(r'Log plot of $10^{x}$')
plt.grid(visible=True,which='both',axis='both')

ax3 = fig.add_subplot(2,2,3)
ax3.plot(x,z,color = 'green', lw = lw)
ax3.set_yscale('log')
ax3.set_font = 20
ax3.set_title(r'Log plot of $2^{x}$')
plt.grid(visible=True,which='both',axis='both')

ax4 = fig.add_subplot(2,2,4)
ax4.plot(x,z,color = 'c', lw = lw)
ax4.set_yscale('linear')
ax4.set_font = 20
ax4.set_title(r'Log plot of $2^{x}$')
plt.grid(visible=True,which='both',axis='both')

fig.tight_layout()
plt.show()

#== Pie chart
#====


# ===================
# Dataset cleaning
# ==================
# %%
import pandas as pd
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/main/nba.csv'
df = pd.read_csv(url)
df['Salary'].fillna(df['Salary'].mean(), inplace=True)
df['College'].fillna(df['College'].mode()[0], inplace=True)
k = df[df['Name'].isna()==True].index
df.drop(k, inplace=True)
print(df.isnull().sum())


df3 = df[['Position','Salary']].groupby('Position').mean()
df3 = df3.reset_index()
df3.index = df3['Position']
df3.plot(kind = 'pie', y='Salary', autopct='%1.0f%%')
plt.title('Pie plot of NBA Team vers Salary (Average)')
plt.legend()
plt.tight_layout()
plt.show()
