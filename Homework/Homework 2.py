#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FixedFormatter

#%%
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/CONVENIENT_global_confirmed_cases.csv'

#%%
#Q1

df = pd.read_csv(url)
df = df.dropna()

#print(df.isna().sum())

#%%
#Q2

df['China_sum'] = df.filter(like='China.').apply(pd.to_numeric, errors='coerce').sum(axis=1)

#%%
#Q3

df['United Kingdom_sum'] = df.filter(like='United Kingdom.').apply(pd.to_numeric, errors='coerce').sum(axis=1)

#%%
#Q4 - US
df['date'] = pd.to_datetime(df['Country/Region'], format='%m/%d/%y')

plt.figure(figsize=(10, 8))
plt.plot(df['date'], df['US'], label="US")

plt.xlabel("Year")
plt.ylabel("Confirmed COVID-19 cases")
plt.title("US confirmed COVID-19 cases")
plt.grid(True)
plt.legend(loc='upper right')


ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())

unique_months = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='MS')

labels = [d.strftime('%b\n%Y') if d.month == 2 and d.year == 2020 else d.strftime('%b') for d in unique_months]

ax.xaxis.set_major_locator(FixedLocator(unique_months.map(mdates.date2num)))
ax.xaxis.set_major_formatter(FixedFormatter(labels))

ax.xaxis.set_minor_locator(mdates.WeekdayLocator())  # Set minor ticks at weekly intervals
ax.tick_params(axis='x', which='minor', length=4, color='gray')

plt.xticks(rotation=0, ha='center')
plt.xlim(df['date'].min(), df['date'].max())
plt.show()

#%%
#Q5 - UK

plt.figure(figsize=(10, 8))
plt.plot(df['date'], df['United Kingdom_sum'], label="UK")

plt.xlabel("Year")
plt.ylabel("Confirmed COVID-19 cases")
plt.title("United Kingdom confirmed COVID-19 cases")
plt.grid(True)
plt.legend(loc='upper right')


ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())

unique_months = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='MS')

labels = [d.strftime('%b\n%Y') if d.month == 2 and d.year == 2020 else d.strftime('%b') for d in unique_months]

ax.xaxis.set_major_locator(FixedLocator(unique_months.map(mdates.date2num)))
ax.xaxis.set_major_formatter(FixedFormatter(labels))

ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
ax.tick_params(axis='x', which='minor', length=4, color='gray')

plt.xticks(rotation=0, ha='center')
plt.xlim(df['date'].min(), df['date'].max())
plt.show()

#%%
#Q5 - China

plt.figure(figsize=(10, 8))
plt.plot(df['date'], df['China_sum'], label="China")

plt.xlabel("Year")
plt.ylabel("Confirmed COVID-19 cases")
plt.title("China confirmed COVID-19 cases")
plt.grid(True)
plt.legend(loc='upper right')


ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())

unique_months = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='MS')

labels = [d.strftime('%b\n%Y') if d.month == 2 and d.year == 2020 else d.strftime('%b') for d in unique_months]

ax.xaxis.set_major_locator(FixedLocator(unique_months.map(mdates.date2num)))
ax.xaxis.set_major_formatter(FixedFormatter(labels))

ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
ax.tick_params(axis='x', which='minor', length=4, color='gray')

plt.xticks(rotation=0, ha='center')
plt.xlim(df['date'].min(), df['date'].max())
plt.show()

#%%
#Q5 - Germany

plt.figure(figsize=(10, 8))
plt.plot(df['date'], df['Germany'], label="Germany")

plt.xlabel("Year")
plt.ylabel("Confirmed COVID-19 cases")
plt.title("Germany confirmed COVID-19 cases")
plt.grid(True)
plt.legend(loc='upper right')


ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())

unique_months = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='MS')

labels = [d.strftime('%b\n%Y') if d.month == 2 and d.year == 2020 else d.strftime('%b') for d in unique_months]

ax.xaxis.set_major_locator(FixedLocator(unique_months.map(mdates.date2num)))
ax.xaxis.set_major_formatter(FixedFormatter(labels))

ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
ax.tick_params(axis='x', which='minor', length=4, color='gray')

plt.xticks(rotation=0, ha='center')
plt.xlim(df['date'].min(), df['date'].max())
plt.show()

#%%
#Q5 - Brazil

plt.figure(figsize=(10, 8))
plt.plot(df['date'], df['Brazil'], label="Brazil")

plt.xlabel("Year")
plt.ylabel("Confirmed COVID-19 cases")
plt.title("Brazil confirmed COVID-19 cases")
plt.grid(True)
plt.legend(loc='upper right')


ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())

unique_months = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='MS')

labels = [d.strftime('%b\n%Y') if d.month == 2 and d.year == 2020 else d.strftime('%b') for d in unique_months]

ax.xaxis.set_major_locator(FixedLocator(unique_months.map(mdates.date2num)))
ax.xaxis.set_major_formatter(FixedFormatter(labels))

ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
ax.tick_params(axis='x', which='minor', length=4, color='gray')

plt.xticks(rotation=0, ha='center')
plt.xlim(df['date'].min(), df['date'].max())
plt.show()

#%%
#Q5 - India

plt.figure(figsize=(10, 8))
plt.plot(df['date'], df['India'], label="India")

plt.xlabel("Year")
plt.ylabel("Confirmed COVID-19 cases")
plt.title("India confirmed COVID-19 cases")
plt.grid(True)
plt.legend(loc='upper right')


ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())

unique_months = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='MS')

labels = [d.strftime('%b\n%Y') if d.month == 2 and d.year == 2020 else d.strftime('%b') for d in unique_months]

ax.xaxis.set_major_locator(FixedLocator(unique_months.map(mdates.date2num)))
ax.xaxis.set_major_formatter(FixedFormatter(labels))

ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
ax.tick_params(axis='x', which='minor', length=4, color='gray')

plt.xticks(rotation=0, ha='center')
plt.xlim(df['date'].min(), df['date'].max())
plt.show()

#%%
#Q5 - Italy

plt.figure(figsize=(10, 8))
plt.plot(df['date'], df['Italy'], label="Italy")

plt.xlabel("Year")
plt.ylabel("Confirmed COVID-19 cases")
plt.title("Italy confirmed COVID-19 cases")
plt.grid(True)
plt.legend(loc='upper right')


ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())

unique_months = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='MS')

labels = [d.strftime('%b\n%Y') if d.month == 2 and d.year == 2020 else d.strftime('%b') for d in unique_months]

ax.xaxis.set_major_locator(FixedLocator(unique_months.map(mdates.date2num)))
ax.xaxis.set_major_formatter(FixedFormatter(labels))

ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
ax.tick_params(axis='x', which='minor', length=4, color='gray')

plt.xticks(rotation=0, ha='center')
plt.xlim(df['date'].min(), df['date'].max())
plt.show()

#%%
#Q6
countries = ['United Kingdom_sum', 'China_sum', 'US', 'Italy', 'Brazil', 'Germany', 'India']

plt.figure(figsize=(10, 8))
plt.plot(df['date'], df[countries], label=countries)

plt.xlabel("Year")
plt.ylabel("Confirmed COVID-19 cases")
plt.title("Global confirmed COVID-19 cases")
plt.grid(True)
plt.legend(loc='upper left')


ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())

unique_months = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='MS')

labels = [d.strftime('%b\n%Y') if d.month == 2 and d.year == 2020 else d.strftime('%b') for d in unique_months]

ax.xaxis.set_major_locator(FixedLocator(unique_months.map(mdates.date2num)))
ax.xaxis.set_major_formatter(FixedFormatter(labels))

ax.xaxis.set_minor_locator(mdates.WeekdayLocator())
ax.tick_params(axis='x', which='minor', length=4, color='gray')

plt.xticks(rotation=0, ha='center')
plt.xlim(df['date'].min(), df['date'].max())
plt.show()

#%%
#Q7
countries = ['United Kingdom_sum', 'China_sum', 'Italy', 'Brazil', 'Germany', 'India']

fig, axes = plt.subplots(3, 2, figsize=(12, 8))

for ax, country in zip(axes.flatten(), countries):
    country_data = df[country]
    ax.bar(df['date'], country_data, width=5)
    ax.set_title(f"{country} confirmed COVID19 cases")
    ax.set_xlabel("Date")
    ax.set_ylabel("Confirmed COVID19 cases")
    ax.grid(True)
    ax.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

#%%
#Q8 - calculate mean, variance, median by column country
countries = ['China_sum', 'United Kingdom_sum', 'Italy', 'Brazil', 'Germany', 'India']

mean_values = df[countries].mean(axis=0).round(2)
variance_values = df[countries].var(axis=0, ddof=1).round(2)
median_values = df[countries].median(axis=0).round(2)

df_countries = pd.DataFrame({
    'Country': countries,
    'Mean': mean_values.values,
    'Variance': variance_values.values,
    'Median': median_values.values
})

print(df_countries)

#%%
#Titanic
import seaborn as sns

titanic = sns.load_dataset('titanic')
titanic.dropna(inplace=True)
print(titanic.isna().sum())
print(titanic.head())

#%%
#Q2

gender_counts = titanic["sex"].value_counts()
print("Number of males:", gender_counts["male"])
print("Number of females:", gender_counts["female"])

plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=[f"{label} ({count})" for label, count in zip(gender_counts.index, gender_counts)], startangle=90)
plt.title("Distribution of Males and Females on Titanic")
plt.show()

#%%
#Q3
total_passengers = gender_counts.sum()
male_percentage = (gender_counts["male"] / total_passengers) * 100
female_percentage = (gender_counts["female"] / total_passengers) * 100

print(f"Percentage of males: {male_percentage:.1f}%")
print(f"Percentage of females: {female_percentage:.1f}%")

plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of Males and Females on Titanic")
plt.show()

#%%
#Q4

males = titanic[titanic['sex'] == 'male']
survived_count = males['alive'].value_counts()

print(f"Males who survived: {survived_count.get('yes', 0)}")
print(f"Males who did not survive: {survived_count.get('no', 0)}")

labels = ['Survived', 'Did not survive']
sizes = [survived_count.get('yes', 0), survived_count.get('no', 0)]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
plt.title('Survival Rate of Males on Titanic')
plt.axis('equal')
plt.show()

#%%
#Q5
females = titanic[titanic['sex'] == 'female']
survived_count = females['alive'].value_counts()

print(f"Female who survived: {survived_count.get('yes', 0)}")
print(f"Females who did not survive: {survived_count.get('no', 0)}")

labels = ['Survived', 'Did not survive']
sizes = [survived_count.get('yes', 0), survived_count.get('no', 0)]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
plt.title('Survival Rate of Females on Titanic')
plt.axis('equal')
plt.show()

#%%
#Q6
class_counts = titanic['pclass'].value_counts()

print(f"First class passengers: {class_counts.get(1, 0)}")
print(f"Second class passengers: {class_counts.get(2, 0)}")
print(f"Third class passengers: {class_counts.get(3, 0)}")


labels = ['First Class', 'Second Class', 'Third Class']
sizes = [class_counts.get(1, 0), class_counts.get(2, 0), class_counts.get(3, 0)]
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ffcc99', '#66b3ff', '#ff6666'])
plt.title('Class Distribution of Passengers on Titanic')
plt.axis('equal')
plt.show()




#%%
#Q7
class_survival = titanic.groupby(['pclass', 'alive']).size().unstack(fill_value=0)

class_survival['survival_percentage'] = (class_survival['yes'] / (class_survival['yes'] + class_survival['no'])) * 100
total_survivors = class_survival['yes'].sum()
class_survival['survival_percentage_among_classes'] = (class_survival['yes'] / total_survivors) * 100

for cls in class_survival.index:
    survival_rate_among_classes = class_survival.loc[cls, 'survival_percentage_among_classes']
    print(f"Class {cls} - Survival Percentage Among Three Classes: {survival_rate_among_classes:.1f}%")

labels = [f'Class {cls}' for cls in class_survival.index]
sizes = class_survival['survival_percentage_among_classes'].values
colors = ['#ffcc99', '#66b3ff', '#ff6666']


fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
ax.set_title('Survival Percentage Rate Based on Ticket Class')
ax.axis('equal')
plt.show()
#%%
#Q8
class_survival = titanic.groupby(['pclass', 'alive']).size().unstack(fill_value=0)

for class_num in class_survival.index:
    survived = class_survival.loc[class_num, 'yes']
    not_survived = class_survival.loc[class_num, 'no']
    total_passengers = survived + not_survived
    survival_rate = (survived / total_passengers) * 100
    print(f"Class {class_num} - Survived: {survived}, Did not survive: {not_survived}, Survival Rate: {survival_rate:.2f}%")


fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for i, class_num in enumerate([1, 2, 3]):
    survived_count = class_survival.loc[class_num, 'yes']
    not_survived_count = class_survival.loc[class_num, 'no']
    sizes = [survived_count, not_survived_count]
    labels = ['Survived', 'Did not survive']

    axes[i].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
    axes[i].set_title(f'Class {class_num} Survival')

plt.tight_layout()
plt.show()

#%%
#Q9
fig, axes = plt.subplots(3, 3, figsize=(16, 8))

def plot_gender_distribution(ax):
    gender_counts = titanic["sex"].value_counts()

    ax.pie(gender_counts, labels=[f"{label} ({count})" for label, count in zip(gender_counts.index, gender_counts)], startangle=90)
    ax.set_title("Distribution of Males and Females on Titanic")

def plot_gender_pct_distribution(ax):
    gender_counts = titanic["sex"].value_counts()

    ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90)
    ax.set_title("Distribution of Males and Females on Titanic")

def plot_male_survival(ax):
    males = titanic[titanic['sex'] == 'male']
    survived_count = males['alive'].value_counts()

    labels = ['Survived', 'Did not survive']
    sizes = [survived_count.get('yes', 0), survived_count.get('no', 0)]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
    ax.set_title('Survival Rate of Males')

def plot_female_survival(ax):
    females = titanic[titanic['sex'] == 'female']
    survived_count = females['alive'].value_counts()

    labels = ['Survived', 'Did not survive']
    sizes = [survived_count.get('yes', 0), survived_count.get('no', 0)]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
    ax.set_title('Survival Rate of Females')

def plot_class_distribution(ax):
    class_counts = titanic['pclass'].value_counts()
    labels = ['Class 1', 'Class 2', 'Class 3']
    sizes = [class_counts.get(1, 0), class_counts.get(2, 0), class_counts.get(3, 0)]
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#ffcc99', '#66b3ff', '#ff6666'])
    ax.set_title('Class Distribution of Passengers on Titanic')

def plot_survival_rate_among_classes(ax):
    class_survival = titanic.groupby(['pclass', 'alive']).size().unstack(fill_value=0)
    class_survival['survival_percentage'] = (class_survival['yes'] / (class_survival['yes'] + class_survival['no'])) * 100
    total_survivors = class_survival['yes'].sum()
    class_survival['survival_percentage_among_classes'] = (class_survival['yes'] / total_survivors) * 100

    labels = [f'Class {cls}' for cls in class_survival.index]
    sizes = class_survival['survival_percentage_among_classes'].values
    colors = ['#ffcc99', '#66b3ff', '#ff6666']
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.set_title('Survival Percentage Rate Based on Ticket Class')


class_survival = titanic.groupby(['pclass', 'alive']).size().unstack(fill_value=0)
for i, class_num in enumerate([1, 2, 3]):
    survived_count = class_survival.loc[class_num, 'yes']
    not_survived_count = class_survival.loc[class_num, 'no']

    sizes = [survived_count, not_survived_count]
    labels = ['Survived', 'Did not survive']

    axes[2, i].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#ff6666'])
    axes[2, i].set_title(f'Class {class_num} Survival')


plot_gender_distribution(axes[0, 0])
plot_gender_pct_distribution(axes[0, 1])
plot_male_survival(axes[1, 0])
plot_female_survival(axes[1, 1])
plot_class_distribution(axes[0, 2])
plot_survival_rate_among_classes(axes[1, 2])


plt.tight_layout()
plt.show()
