#%%
from sklearn.datasets import fetch_lfw_people
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#%%

faces = fetch_lfw_people(min_faces_per_person=20)

X = faces.data
X1 = pd.DataFrame(X)

y = faces.target
y1 = pd.DataFrame(y, columns = ['label'])

df = pd.concat([X1, y1], axis = 1)

#%%
sns.set_style('whitegrid')
sns.set(font_scale=1)

plt.figure(figsize = (16, 8))

sns.countplot(y1,
            x="label",
            palette="hls",
            order = y1.label.value_counts(ascending=True).index)

plt.title("Distribution of Labels")
plt.show()

#%%
min_img = int(y1['label'].value_counts().min())
new_y1 = pd.DataFrame({'label': y1['label'].unique(),
                       'count': min_img})

#%%
plt.figure(figsize = (16, 8))
sns.barplot(new_y1,
              x="label",
              y="count",
              palette="hls")

plt.title("Distribution of Balanced Labels")
plt.show()

#%%
sns.set_style('whitegrid')
fig, axes = plt.subplots(2, 1, figsize=(16, 8))

sns.countplot(y1,
            x="label",
            palette="hls",
            order = y1.label.value_counts(ascending=True).index,
            ax=axes[0])

axes[0].set_title("Colors by Category")

sns.barplot(new_y1,
            x="label",
            y="count",
            palette="hls",
            ax=axes[1])

axes[1].set_title("Balanced Dataset")

plt.tight_layout()
plt.show()