#%%
import pandas as pd
import numpy as np
from sklearn.utils import resample

import seaborn as sns
import matplotlib.pyplot as plt
#%%
url = 'https://raw.githubusercontent.com/wilona-nguyen/Visualization/refs/heads/main/Final%20Poject/thyroid_cancer_risk_data.csv'

df = pd.read_csv(url)
df = df.drop('Patient_ID', axis=1)
print(df.head())



#%%

# Set up the figure and axes for subplots
num_features = len(df.columns)
num_rows = 3
num_cols = 6
fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 12))
axes = axes.flatten()   # Flatten axes to easily iterate through them

# Loop through each column in the DataFrame
for i, feature in enumerate(df.columns):
    ax = axes[i]  # Select the current axis

    # Check if the feature is categorical or numerical
    if df[feature].dtype == 'object' or df[feature].dtype.name == 'category':  # Categorical
        feature_order = df[feature].value_counts().index
        sns.countplot(x=feature, data=df, palette='muted', order=feature_order, ax=ax)

        # Add counts on top of each bar
        for p in ax.patches:
            ax.annotate(f'{int(p.get_height())}',
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center', fontsize=10, color='black',
                        xytext=(0, 5), textcoords='offset points')

        ax.set_title(f'Distribution of {feature}')
        ax.set_xlabel(feature)
        ax.set_ylabel('Count')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    else:  # Numerical
        sns.boxplot(x=df[feature], ax=ax, palette='muted')

        ax.set_title(f'Boxplot of {feature}')
        ax.set_xlabel(feature)



# Adjust layout to make sure everything fits
plt.tight_layout()
plt.show()



#%% - Distribution of each gender in the dataset
sns.set(style="whitegrid")

# Create the count plot
plt.figure(figsize=(10, 7))
sns.countplot(x='Gender', hue = 'Gender', data=df, palette='muted')

# Customize plot
plt.title('Gender Distribution in the Dataset')
plt.xlabel('Gender')
plt.ylabel('Count')

# Show plot
plt.show()

#%% - Country distribution
sns.set(style="whitegrid")

country_order = df['Country'].value_counts().index

# Create the count plot
plt.figure(figsize=(10, 7))
ax = sns.countplot(x='Country', hue = 'Country', data=df, palette='muted', order = country_order)

# Customize plot
plt.title('Country Distribution in the Dataset')
plt.xlabel('Country')
plt.ylabel('Count')

# Add counts on top of each bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

# Show plot
plt.show()

#%% - Ethnicity distribution
sns.set(style="whitegrid")

country_order = df['Ethnicity'].value_counts().index

# Create the count plot
plt.figure(figsize=(10, 7))
ax = sns.countplot(x='Ethnicity', hue = 'Ethnicity', data=df, palette='muted', order = country_order)

# Customize plot
plt.title('Ethnicity Distribution in the Dataset')
plt.xlabel('Ethnicity')
plt.ylabel('Count')

# Add counts on top of each bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

# Show plot
plt.show()


#%% - Gender distribution
sns.set(style="whitegrid")

country_order = df['Gender'].value_counts().index

# Create the count plot
plt.figure(figsize=(10, 7))
ax = sns.countplot(x='Gender', hue = 'Gender', data=df, palette='muted', order = country_order)

# Customize plot
plt.title('Gender Distribution in the Dataset')
plt.xlabel('Gender')
plt.ylabel('Count')

# Add counts on top of each bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

# Show plot
plt.show()

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

#%% - Distribution of thyroid cancer risk in each country

fontsize_label = 20
fontsize_legend = 20
fontsize_title = 25

sns.set(style='whitegrid')
plt.figure(figsize=(20,10))

sns.countplot(data = df,
            y = 'Country',
            hue = 'Thyroid_Cancer_Risk',
            palette='muted',
            order=df['Country'].value_counts(ascending=False).index)

plt.xlabel('Country', fontsize = fontsize_label)
plt.ylabel('Count', fontsize = fontsize_label)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = fontsize_legend)
plt.title('Distribution of Thyroid Cancer Risk in Each Country', fontsize=fontsize_title)
plt.show()


#%% - distribution of thyroid cancer risk for each ethnicity
sns.set(style='whitegrid')
plt.figure(figsize=(20,10))

sns.countplot(data = df,
            y = 'Ethnicity',
            hue = 'Thyroid_Cancer_Risk',
            order=df['Ethnicity'].value_counts(ascending=False).index)

plt.xlabel('Ethnicity', fontsize = fontsize_label)
plt.ylabel('Count', fontsize = fontsize_label)

plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)
plt.legend(fontsize = fontsize_legend)
plt.title('Distribution of Thyroid Cancer Risk for Each Ethnicity', fontsize=fontsize_title)
plt.show()

#%%
#trial pairwise plot, but too much data points, not good

sns.pairplot(df, hue='Diagnosis', diag_kind='kde', corner=True)
plt.suptitle("Pairwise Relationships", y=1.02)
plt.show()

#%%
sns.set(style="whitegrid")

risk_order = {'Low': 0, 'Medium': 1, 'High': 2}
df['Risk_Level'] = df['Thyroid_Cancer_Risk'].map(risk_order)

# Create the boxplot
plt.figure(figsize=(14, 10))
sns.boxplot(x='Gender', y='Risk_Level', hue='Diagnosis', data=df, palette='muted')

# Customize plot
plt.title('Thyroid Cancer Risk by Gender and Diagnosis')
plt.xlabel('Gender')
plt.ylabel('Thyroid Cancer Risk (0 = Low, 1 = Medium, 2 = High)')
plt.xticks(rotation=0)
plt.legend(title='Diagnosis', bbox_to_anchor=(1.05, 1), loc='upper left')

# Show plot
plt.show()


#%% - get df for high risk
df_high_risk = df[df['Thyroid_Cancer_Risk'] == 'High']
df_high_risk.shape

#%% - country distribution in High risk
sns.set(style="whitegrid")

country_order = df_high_risk['Country'].value_counts().index

# Create the count plot
plt.figure(figsize=(10, 7))
ax = sns.countplot(x='Country', hue = 'Country', data=df_high_risk, palette='muted', order = country_order)

# Customize plot
plt.title('Country Distribution in the High Risk Dataset')
plt.xlabel('Country')
plt.ylabel('Count')

# Add counts on top of each bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

# Show plot
plt.show()

#%% - Ethnicity distribution in high risk
sns.set(style="whitegrid")

country_order = df_high_risk['Ethnicity'].value_counts().index

# Create the count plot
plt.figure(figsize=(10, 7))
ax = sns.countplot(x='Ethnicity', hue = 'Ethnicity', data=df_high_risk, palette='muted', order = country_order)

# Customize plot
plt.title('Ethnicity Distribution in the High Risk Dataset')
plt.xlabel('Ethnicity')
plt.ylabel('Count')

# Add counts on top of each bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

# Show plot
plt.show()


#%% - Gender distribution in high risk
sns.set(style="whitegrid")

country_order = df_high_risk['Gender'].value_counts().index

# Create the count plot
plt.figure(figsize=(10, 7))
ax = sns.countplot(x='Gender', hue = 'Gender', data=df_high_risk, palette='muted', order = country_order)

# Customize plot
plt.title('Gender Distribution in the High Risk Dataset')
plt.xlabel('Gender')
plt.ylabel('Count')

# Add counts on top of each bar
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                textcoords='offset points')

# Show plot
plt.show()

#%%
fontsize_label = 12
fontsize_title = 15

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Adjust rows and columns based on your plot count
axes = axes.flatten()  # Flatten the axes to easily iterate

# List of columns to plot
columns = ['Family_History', 'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 'Obesity', 'Diabetes']

# Plot each distribution in a subplot
for i, col in enumerate(columns):
    sns.histplot(df_high_risk[col], ax=axes[i])  # You can use kde=False for only histograms
    axes[i].set_title(f'Distribution of {col}', fontsize=fontsize_title)
    axes[i].set_xlabel(col, fontsize = fontsize_label)
    axes[i].set_ylabel('Frequency', fontsize = fontsize_label)

    axes[i].tick_params(axis='x', labelsize=12)

# Adjust layout
plt.suptitle('High Risk Dataset', fontweight='bold', fontsize=20)
plt.tight_layout()
plt.show()