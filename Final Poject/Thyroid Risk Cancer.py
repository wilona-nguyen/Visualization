#%%
import pandas as pd
import numpy as np
from sklearn.utils import resample

import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
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

#%%
import pandas as pd
import matplotlib.pyplot as plt

# Group by Thyroid_Risk_Cancer and Gender, and count occurrences
grouped = df.groupby(['Thyroid_Cancer_Risk', 'Gender']).size().unstack()

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define the risk categories
risk_categories = ['Low', 'Medium', 'High']

# Create a pie chart for each risk category
for i, risk in enumerate(risk_categories):
    if risk in grouped.index:
        data = grouped.loc[risk]
        axes[i].pie(data,
                    labels=data.index,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=['lightcoral', 'lightblue'],
                    textprops={'fontsize': 15})
        axes[i].set_title(f'{risk} Thyroid Cancer Risk', fontsize = 15)

plt.suptitle('Gender Distribution for Each Thyroid Cancer Risk Level', fontweight = 'bold', fontsize = 20)
plt.subplots_adjust(top=0.2)
plt.tight_layout()
plt.show()


#%% - Country distribution
sns.set_theme(style="whitegrid")

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

#%%

fig = px.sunburst(df, path=['Thyroid_Cancer_Risk', 'Country'], values='Age')
fig.show()

#%% - Distribution of cancer risk level for each country
df_count = df.groupby(['Country', 'Thyroid_Cancer_Risk']).size().unstack(fill_value=0)

# Sort countries by the total count (sum of all risk categories) in descending order
df_count = df_count.loc[df_count.sum(axis=1).sort_values(ascending=False).index]

# Plot stacked bar chart with 'muted' color palette
ax = df_count.plot(kind='bar', stacked=True, figsize=(10, 6))
ax.set_ylabel('Count')
ax.set_title('Thyroid Cancer Risk by Country')
plt.xticks(rotation=30)
plt.show()


#%% - Age distribution

sns.histplot(df['Age'], bins=20)
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


#%%

plt.figure(figsize=(10, 6))
sns.boxplot(x='Thyroid_Cancer_Risk', y='Age', data=df)
plt.title('Age Distribution by Thyroid Cancer Risk')
plt.xlabel('Thyroid Cancer Risk')
plt.ylabel('Age')
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

import squarify
import matplotlib.pyplot as plt
import seaborn as sns

# Get country counts
country_counts = df_high_risk['Country'].value_counts()

# Define colors using a seaborn palette
colors = sns.color_palette("muted", len(country_counts))

# Create labels with country names and counts
labels = [f'{country}\n{count}' for country, count in zip(country_counts.index, country_counts.values)]

# Create the treemap
plt.figure(figsize=(14, 10))
squarify.plot(
    sizes=country_counts.values,
    label=labels,
    color=colors,
    alpha=0.7,
    text_kwargs={'fontsize': 15, 'weight': 'bold'}
)

# Add borders by overlaying white lines
norm_sizes = squarify.normalize_sizes(country_counts.values, 100, 100)
rects = squarify.squarify(norm_sizes, 0, 0, 100, 100)
for rect in rects:
    plt.gca().add_patch(plt.Rectangle((rect['x'], rect['y']), rect['dx'], rect['dy'],
                                      edgecolor='white', linewidth=3, fill=False))

# Customize plot
plt.title('Country Distribution in the High Risk Dataset', fontsize=20, fontweight = 'bold', pad = 30)
plt.axis('off')  # Hide axis

# Show plot
plt.show()



#%% - Ethnicity distribution in high risk
sns.set(style="whitegrid")

country_order = df_high_risk['Ethnicity'].value_counts().index

# Create the count plot
plt.figure(figsize=(10, 7))
ax = sns.countplot(x='Ethnicity', hue = 'Ethnicity', data=df_high_risk, palette='muted', order = country_order)

# Customize plot
plt.title('Ethnicity Distribution in the High Risk Dataset', fontweight = 'bold', fontsize = 20)
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

#%% - Health background distribution
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Define your label sizes
fontsize_label = 15
fontsize_title = 15

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Adjust rows and columns based on your plot count
axes = axes.flatten()  # Flatten the axes to easily iterate

# List of columns to plot
columns = ['Family_History', 'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 'Obesity', 'Diabetes']

# Define the order for the 'Yes' and 'No' categories
yes_no_order = ['Yes', 'No']

# Ensure each column is categorical with the desired order
for col in columns:
    df_high_risk[col] = pd.Categorical(df_high_risk[col], categories=yes_no_order, ordered=True)

# Plot each distribution in a subplot
for i, col in enumerate(columns):
    sns.histplot(df_high_risk[col], ax=axes[i], discrete=True)
    axes[i].set_title(f'Distribution of {col}', fontsize=fontsize_title)
    axes[i].set_xlabel(col, fontsize=fontsize_label)
    axes[i].set_ylabel('Frequency', fontsize=fontsize_label)

    axes[i].tick_params(axis='x', labelsize=12)

# Adjust layout
plt.suptitle('High Risk Dataset', fontweight='bold', fontsize=20)
plt.tight_layout()
plt.show()


#%% - distribution of TSH Level in each risk level



fontsize_label = 15
fontsize_legend = 10
fontsize_title = 20

sns.set(style='whitegrid')

# Create box plot
plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='Thyroid_Cancer_Risk', y='TSH_Level', hue = 'Thyroid_Cancer_Risk')

plt.axhspan(0, 0.4, color='red', alpha=0.2, label='Overactive Thyroid (TSH < 0.4)')  # Shaded area for overactive thyroid
plt.axhspan(0.4, 4.0, color='green', alpha=0.2, label='Normal TSH Range (0.4 - 4.0)')  # Shaded area for normal range
plt.axhspan(4.0, df['TSH_Level'].max(), color='orange', alpha=0.2, label='Underactive Thyroid (TSH > 4.0)')  # Shaded area for underactive thyroid

# Customize labels and title
plt.xlabel('Thyroid Cancer Risk', fontsize=12)
plt.ylabel('TSH Level', fontsize=12)
plt.title('Boxplot of TSH Level by Thyroid Cancer Risk', fontsize=15)

plt.legend(title='Thyroid Cancer Risk', fontsize=10, title_fontsize=12, loc='upper right')


# Show plot
plt.tight_layout()
plt.show()

#%%

# Create categories based on TSH_Level
conditions = [
    (df['TSH_Level'] < 0.4),    # Overactive thyroid
    (df['TSH_Level'] >= 0.4) & (df['TSH_Level'] <= 4.0),  # Normal range
    (df['TSH_Level'] > 4.0)     # Underactive thyroid
]

choices = ['Overactive Thyroid', 'Normal TSH Range', 'Underactive Thyroid']

# Create a new column with the categories
df['TSH_Category'] = np.select(conditions, choices, default='Normal')

# Count the occurrences in each category
category_counts = df['TSH_Category'].value_counts()

# Plotting the pie chart
plt.figure(figsize=(8, 8))
plt.pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'], startangle=90)
plt.title('Distribution of TSH Levels')
plt.axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
plt.show()


#%%


# Create categories based on TSH_Level
conditions = [
    (df['TSH_Level'] < 0.4),    # Overactive thyroid
    (df['TSH_Level'] >= 0.4) & (df['TSH_Level'] <= 4.0),  # Normal range
    (df['TSH_Level'] > 4.0)     # Underactive thyroid
]

choices = ['Overactive Thyroid', 'Normal TSH Range', 'Underactive Thyroid']

# Create a new column with the categories
df['TSH_Category'] = np.select(conditions, choices, default='Normal')

# Count the occurrences in each TSH Category
category_counts = df['TSH_Category'].value_counts()

# Create a new column based on Thyroid Cancer Risk for each TSH category
df['TSH_Cancer_Risk_Distribution'] = df.groupby('TSH_Category')['Thyroid_Cancer_Risk'].transform(lambda x: x.value_counts(normalize=True))

# Create a count of Thyroid Cancer Risk distribution for each TSH Category
cancer_risk_counts = df.groupby(['TSH_Category', 'Thyroid_Cancer_Risk']).size().unstack(fill_value=0)

# Plotting the dual pie charts
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# First Pie Chart for TSH Level Distribution
axes[0].pie(category_counts, labels=category_counts.index, autopct='%1.1f%%', colors=['#ff9999', '#66b3ff', '#99ff99'], startangle=90)
axes[0].set_title('Distribution of TSH Levels', fontsize=14)
axes[0].axis('equal')  # Equal aspect ratio ensures the pie chart is circular.

# Second Pie Chart for Thyroid Cancer Risk Distribution within each TSH Category
for i, category in enumerate(category_counts.index):
    axes[1].pie(cancer_risk_counts.loc[category], labels=cancer_risk_counts.columns, autopct='%1.1f%%',
                startangle=90, radius=0.75, colors=['#ffcccc', '#ffcc99', '#99ffcc'], wedgeprops=dict(width=0.3))
    axes[1].set_title(f'Thyroid Cancer Risk for {category} TSH Levels', fontsize=12)

# Show the plot
plt.tight_layout()
plt.show()


#%% - Dsitribution of High, Medium, Low risk

import seaborn as sns
import matplotlib.pyplot as plt

# Bar plot for the distribution of Thyroid_Cancer_Risk
plt.figure(figsize=(8, 6))
sns.countplot(x='Thyroid_Cancer_Risk', data=df, palette='muted')

plt.title('Distribution of Thyroid Cancer Risk')
plt.xlabel('Thyroid Cancer Risk')
plt.ylabel('Count')
plt.show()


#%%-getiinf data on T3 adn T4 level

df_low_risk = df[df['Thyroid_Cancer_Risk'] == 'Low']

#%%
import seaborn as sns
import matplotlib.pyplot as plt

# Define your label sizes
fontsize_label = 15
fontsize_title = 15

# Create subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # Adjust rows and columns based on your plot count
axes = axes.flatten()  # Flatten the axes to easily iterate

# List of columns to plot
columns = ['Family_History', 'Radiation_Exposure', 'Iodine_Deficiency', 'Smoking', 'Obesity', 'Diabetes']

# Define the order for the 'Yes' and 'No' categories
yes_no_order = ['Yes', 'No']

# Ensure each column is categorical with the desired order
for col in columns:
    df_low_risk[col] = pd.Categorical(df_low_risk[col], categories=yes_no_order, ordered=True)

# Plot each distribution in a subplot
for i, col in enumerate(columns):
    sns.histplot(df_low_risk[col], ax=axes[i], discrete=True)
    axes[i].set_title(f'Distribution of {col}', fontsize=fontsize_title)
    axes[i].set_xlabel(col, fontsize=fontsize_label)
    axes[i].set_ylabel('Frequency', fontsize=fontsize_label)
    axes[i].tick_params(axis='x', labelsize=12)

# Adjust layout
plt.suptitle('Low Risk Dataset', fontweight='bold', fontsize=20)
plt.tight_layout()
plt.show()


#%% - Heatmap

import seaborn as sns
import matplotlib.pyplot as plt

# Assuming df has columns: 'TSH_Level', 'T3_Level', and 'T4_Level'

# Calculate the correlation matrix
correlation_matrix = df[['TSH_Level', 'T3_Level', 'T4_Level', 'Nodule_Size']].corr()

# Create the heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.5f',
            cbar=True, square=True, linewidths=0.5, linecolor='black')

# Title of the plot
plt.title('Correlation Heatmap of TSH_Level, T3_Level, T4_Level, and Nodule_Size', fontsize=15, pad=15)

# Display the plot
plt.show()



