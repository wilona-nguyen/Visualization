#%% -- still in the pandas lecture
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/mnist_test.csv'

df = pd.read_csv(url)
print(df.shape)

#%%
pic = df[0:1].values.reshape(785)[1:].reshape(28,28)

plt.figure()
plt.imshow(pic)
plt.show()

#%%
zero = df[df['label'] == 0]

pic = zero[0:1].values.reshape(785)[1:].reshape(28,28)

plt.figure()
plt.imshow(pic)
plt.show()

#%% First 25 pictures of label = 0
zero_images = df[df['label'] == 0].head(25)  # Select the first 25 rows where the label is 0

plt.figure(figsize=(10, 10))  # Set the figure size
for i, (_, row) in enumerate(zero_images.iterrows(), start=1):
    pic = row.values.reshape(785)[1:].reshape(28, 28)  # Convert row to picture
    ax = plt.subplot(5, 5, i)  # Create a subplot in a 5x5 grid
    ax.axis('off')  # Remove axis
    ax.imshow(pic)  # Display the image

plt.tight_layout()
plt.show()

#%% - Professor's solution
plt.figure(figsize=(12, 12))
k = 0
for i in range(5):
    for j in range (5):
        k += 1
        plt.subplot(5, 5, k)
        pic = zero.iloc[k-1,1:].values.reshape(28,28)
        plt.imshow(pic)

plt.tight_layout()
plt.show()

#%% -- Professor's solution
df_agg = df.groupby('label').count().sort_values(by = '1x1', ascending = False)
df_agg.reset_index(inplace = True)

ax = df_agg.iloc[:, 1:2].plot(kind = 'bar')
ax.set_xticklabels(df_agg['label'].values)

plt.tight_layout()
plt.show()

#%% - Load left_wing_politician
from sklearn.datasets import fetch_lfw_people

faces = fetch_lfw_people(min_faces_per_person=20)
print(f'Image shape {faces.images.shape}')
print(f'Number of features {faces.data.shape[1]}')
print(f'Number of classes {faces.target_names}')

#%% -- plot only the 12th person
plt.figure(figsize=(10, 10))
ax = plt.subplot(1, 1, 1)
ax.imshow(faces.images[11])
ax.axis('off')
plt.show()

#%% - Professor's solution
X = faces.data
X1 = pd.DataFrame(X)

y = faces.target
y1 = pd.DataFrame(y, columns = ['label'])

df = pd.concat([X1, y1], axis = 1)

row_12 = df[df['label'] == 12]
pic = row_12.iloc[0, :-1].values.reshape(62,47)
plt.figure(figsize = (8,10))
plt.imshow(pic)
plt.show()

#%% - plot the first 20 observations layout = (5,4) using plt.subplot
plt.figure(figsize=(10, 10))
for i in range(20):
    ax = plt.subplot(5, 4, i + 1)
    ax.imshow(faces.images[i])
plt.show()

#%% - Professor's solution
for i in range(20):
    pic = df.iloc[i, :-1].values.reshape(62,47) #values is to get the numpy array
    plt.subplot(5, 4, i + 1)
    plt.imshow(pic)

plt.show()

#%% - plot in seperate 4 subplots f(x)= x-squared, x-cubed, sqrt of x, and sqrt cubed of x
x = np.linspace(-10, 10, 100)
x = np.array(x, dtype = np.complex64)
y1 = x**2
y2 = x**(1/2)
y3 = x**3
y4 = x**(1/3)

lw = 3
font1 = {'family' : 'serif', 'color' : 'blue', 'size' : 15}
font2 = {'family' : 'serif', 'color' : 'darkred', 'size' : 15}

fig, ax = plt.subplots(2,2, figsize = (10,10))

#x^2
ax[0,0].plot(x, y1, label = '$f(x) = x^2$', lw = lw)
ax[0,0].legend(loc = 'center', fontsize = 15, title_fontsize = 15)
ax[0,0].set_title('$f(x) = x^2$', fontdict = font1)
ax[0,0].set_xlabel('Samples', fontdict = font2)
ax[0,0].set_ylabel('Mag.', fontdict = font2)
ax[0,0].grid()

#x^(1/2)
ax[0,1].plot(x, y2, label = '$f(x) = \sqrt{x}$', lw = lw)
ax[0,1].legend(loc = 'center', fontsize = 15, title_fontsize = 15)
ax[0,1].set_title('$f(x) = \sqrt{x}$', fontdict = font1)
ax[0,1].set_xlabel('Samples', fontdict = font2)
ax[0,1].set_ylabel('Mag.', fontdict = font2)
ax[0,1].grid()

ax[1,0].plot(x, y3, label = '$f(x) = x^3$', lw = lw)
ax[1,0].legend(loc = 'center', fontsize = 15, title_fontsize = 15)
ax[1,0].set_title('$f(x) = x^3$', fontdict = font1)
ax[1,0].set_xlabel('Samples', fontdict = font2)
ax[1,0].set_ylabel('Mag.', fontdict = font2)
ax[1,0].grid()

ax[1,1].plot(x, y4, label = '$f(x) = \sqrt[3]{x}$', lw = lw)
ax[1,1].legend(loc = 'center', fontsize = 15, title_fontsize = 15)
ax[1,1].set_title('$f(x) = \sqrt[3]{x}$', fontdict = font1)
ax[1,1].set_xlabel('Samples', fontdict = font2)
ax[1,1].set_ylabel('Mag.', fontdict = font2)
ax[1,1].grid()

fig.tight_layout()
plt.show()

#%% - Mathematical Functions using plt.fig_addsubplot
x = np.linspace(1, 10)
y = [10 ** el for el in x]
z = [2 ** el for el in x]

fig = plt.figure(figsize = (10,8))

ax1 = fig.add_subplot(2, 2, 1)
ax1.plot(x, y, color = 'b', lw = lw)
ax1.set_yscale('log')
ax1.set_font = 20
ax1.set_title(r'Log plot of $10^{x}$')
plt.grid(visible = True, which = 'both', axis = 'both')

ax2 = fig.add_subplot(2, 2, 2)
ax2.plot(x, y, color = 'r', lw = lw)
ax2.set_yscale('log')
ax2.set_font = 20
ax2.set_title(r'Log plot of $10^{x}$')
plt.grid(visible = True, which = 'both', axis = 'both')

ax3 = fig.add_subplot(2, 2, 3)
ax3.plot(x, z, color = 'g', lw = lw)
ax3.set_yscale('log')
ax3.set_font = 20
ax3.set_title(r'Log plot of $2^{x}$')
plt.grid(visible = True, which = 'both', axis = 'both')

ax4 = fig.add_subplot(2, 2, 4)
ax4.plot(x, z, color = 'c', lw = lw)
ax4.set_yscale('linear')
ax4.set_font = 20
ax4.set_title(r'Log plot of $2^{x}$')
plt.grid(visible = True, which = 'both', axis = 'both')

fig.tight_layout()
plt.show()

#%% - Pie Chart
url ='https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/nba.csv'
df = pd.read_csv(url)
print(df.head())

