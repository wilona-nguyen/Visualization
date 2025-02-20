#%%
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
#%%
url = 'https://raw.githubusercontent.com/wilona-nguyen/Visualization/refs/heads/main/Final%20Poject/thyroid_cancer_risk_data.csv?token=GHSAT0AAAAAAC5I5HOOY2OG6PORFL64CWB6Z5XQXYA'

df = pd.read_csv(url)
print(df.head())

#%%
sns.set(style='whitegrid')
plt.figure(figsize=(20,10))

sns.countplot(data = df,
            x = 'Gender',
            hue = 'Thyroid_Cancer_Risk',
            order=df['Gender'].value_counts(ascending=False).index)

plt.xlabel('Gender', fontsize = 15)
plt.ylabel('Count', fontsize = 15)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 15)
plt.title('Distribution of Thyroid Cancer Risk for Each Gender', fontsize=20)
plt.show()

#%%

sns.set(style='darkgrid')
plt.figure(figsize=(20,10))

sns.countplot(data = df,
            x = 'Country',
            hue = 'Thyroid_Cancer_Risk',
            order=df['Country'].value_counts(ascending=False).index)

plt.xlabel('Country', fontsize = 15)
plt.ylabel('Count', fontsize = 15)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 15)
plt.title('Distribution of Thyroid Cancer Risk in Each Country', fontsize=20)
plt.show()


#%%
sns.set(style='darkgrid')
plt.figure(figsize=(20,10))

sns.countplot(data = df,
            x = 'Ethnicity',
            hue = 'Thyroid_Cancer_Risk',
            order=df['Ethnicity'].value_counts(ascending=False).index)

plt.xlabel('Ethnicity', fontsize = 15)
plt.ylabel('Count', fontsize = 15)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = 15)
plt.title('Distribution of Thyroid Cancer Risk for Each Ethnicity', fontsize=20)
plt.show()
