#%%

from sklearn.datasets import fetch_lfw_people
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#%%
faces = fetch_lfw_people(min_faces_per_person=20)



#%%
X = faces.data
X1 = pd.DataFrame(X)

y = faces.target
y1 = pd.DataFrame(y, columns = ['label'])

df = pd.concat([X1, y1], axis = 1)

#%% - 1. plot ditribution of 'Label' in ascending order

sns.set_style('whitegrid')
plt.figure(figsize=(16,8))

sns.barplot(df,
            y= 'label',
            hue = 'label',
            hue_order=df.label.value_counts().index,
            palette = 'hls',
            font_scale = 1
            )

plt.xlabel('label')
plt.ylabel('count')
plt.title('Distribution of Target Classes')
plt.show()

#%% - 2.down sampling to minimum number of images per politcian



