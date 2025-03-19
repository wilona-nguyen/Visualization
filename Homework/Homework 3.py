#%%
import seaborn as sns
import matplotlib.pyplot as plt


#%% - Q1
df = sns.load_dataset('penguins')
print(df.tail(5).to_string())
df.describe()

#%% - Q2
df.isnull().sum()
df.isna().sum()
df = df.dropna()

#%% - Q3
sns.set_style('darkgrid')

sns.histplot(data=df, x='flipper_length_mm', kde=True)

plt.title('Question 3')
plt.show()

#%% - Q4

sns.set_style('darkgrid')

sns.histplot(data=df, x='flipper_length_mm', kde=True, binwidth=3)

plt.title('Question 4')
plt.show()

#%% - Q5

sns.set_style('darkgrid')

sns.histplot(data=df, x='flipper_length_mm', kde=True,
             binwidth=3,
             bins=30)

plt.title('Question 5')
plt.show()

#%% - Q6


sns.displot(data=df, x='flipper_length_mm', hue = 'species')


plt.title('Question 6')
plt.tight_layout()
plt.show()

#%% - Q7

sns.set_style('white')
plt.figure(figsize=(10, 10))

sns.displot(data=df, x='flipper_length_mm', hue = 'species',
            element = 'step')

plt.title('Question 7')
plt.tight_layout()
plt.show()

#%% -  Q8
sns.histplot(data=df, x='flipper_length_mm', hue='species',
             multiple='stack')

plt.title('Question 8')
plt.show()

#%% - Q9
sns.displot(data=df, x='flipper_length_mm', hue='sex',
            kind = 'hist',
            multiple='dodge')

plt.title('Question 9')
plt.tight_layout()
plt.show()

#%% - Q10
g = sns.displot(
    data=df,
    x="flipper_length_mm",
    col="sex",
    kde=False,
)

g.set_axis_labels("Flipper Length (mm)", "Count")
g.set_titles("{col_name}")

plt.suptitle('Question 10')
plt.tight_layout()
plt.show()

#%% - Q11
plt.figure(figsize=(8, 6))
sns.histplot(
    data=df,
    x="flipper_length_mm",
    hue="species",
    stat="density",
    common_norm=False,
    kde=True,
)

plt.xlabel("Flipper Length (mm)")
plt.ylabel("Density")
plt.title("Question 11")

plt.show()

#%% - Q12

plt.figure(figsize=(8, 6))
sns.histplot(
    data=df,
    x="flipper_length_mm",
    hue="sex",
    stat="density",
    kde=True
)

plt.xlabel("Flipper Length (mm)")
plt.ylabel("Density")
plt.title("Question 12")

plt.show()

#%% - Q13
plt.figure(figsize=(8, 6))
sns.histplot(
    data=df,
    x="flipper_length_mm",
    hue="species",
    stat="probability",
    kde=True,
    common_norm=False,

)

plt.xlabel("Flipper Length (mm)")
plt.ylabel("Probability")
plt.title("Question 13")

plt.show()

#%% - Q14

plt.figure(figsize=(8, 6))

sns.displot(
    data=df,
    x="flipper_length_mm",
    hue="species",
    kind="kde",
)

plt.xlabel("Flipper Length (mm)")
plt.ylabel("Density")
plt.title("Question 14")

plt.tight_layout()
plt.show()

#%% - Q15

plt.figure(figsize=(8, 6))

sns.displot(
    data=df,
    x="flipper_length_mm",
    hue="sex",
    kind="kde",
)

plt.xlabel("Flipper Length (mm)")
plt.ylabel("Density")
plt.title("Question 15")

plt.tight_layout()
plt.show()

#%% - Q16

plt.figure(figsize=(8, 6))

sns.displot(
    data=df,
    x="flipper_length_mm",
    hue="species",
    kind="kde",
    multiple = 'stack'
)

plt.xlabel("Flipper Length (mm)")
plt.ylabel("Density")
plt.title("Question 16")

plt.tight_layout()
plt.show()

#%% - Q17

plt.figure(figsize=(8, 6))

sns.displot(
    data=df,
    x="flipper_length_mm",
    hue="sex",
    kind="kde",
    multiple = 'stack'
)

plt.xlabel("Flipper Length (mm)")
plt.ylabel("Density")
plt.title("Question 17")

plt.tight_layout()
plt.show()

#%% - Q18

plt.figure(figsize=(8, 6))

sns.displot(
    data=df,
    x="flipper_length_mm",
    hue="species",
    kind="kde",
    fill = True
)


plt.xlabel("Flipper Length (mm)")
plt.ylabel("Density")
plt.title("Question 18")

plt.tight_layout()
plt.show()

#%% - Q19

plt.figure(figsize=(8, 6))

sns.displot(
    data=df,
    x="flipper_length_mm",
    hue="sex",
    kind="kde",
    fill = True
)

plt.xlabel("Flipper Length (mm)")
plt.ylabel("Density")
plt.title("Question 19")


plt.tight_layout()
plt.show()

#%% - Q20

plt.figure(figsize=(8, 6))
sns.regplot(
    x="bill_length_mm",
    y="bill_depth_mm",
    data=df,
    scatter_kws={'color': 'blue'},
    line_kws={'color': 'red'}
)

plt.xlabel("Bill Length (mm)")
plt.ylabel("Bill Depth (mm)")
plt.title("Question 20")
plt.show()

#%% - Q21


plt.figure(figsize=(8, 6))
sns.countplot(
    x="island",
    hue="species",
    data=df
)

plt.xlabel("Island")
plt.ylabel("Number of Penguins")
plt.title("Question 21")

plt.show()

#%% - Q22


plt.figure(figsize=(8, 6))

sns.countplot(
    x="sex",
    hue="species",
    data=df
)

plt.xlabel("Sex")
plt.ylabel("Number of Penguins")
plt.title("Question 22")

plt.show()


#%% - Q23


plt.figure(figsize=(8, 6))

sns.kdeplot(
    data=df,
    x="bill_length_mm",
    y="bill_depth_mm",
    hue="sex",
    fill=True,
)


plt.xlabel("Bill Length (mm)")
plt.ylabel("Bill Depth (mm)")
plt.title("Question 23")
plt.show()

#%% - Q24

plt.figure(figsize=(8, 6))

sns.kdeplot(
    data=df,
    x="bill_length_mm",
    y="flipper_length_mm",
    hue="sex",
    fill=True,
)

plt.xlabel("Bill Length (mm)")
plt.ylabel("Flipper Length (mm)")
plt.title("Question 24")
plt.show()

#%% - Q25
plt.figure(figsize=(8, 6))

sns.kdeplot(
    data=df,
    x="flipper_length_mm",
    y="bill_depth_mm",
    hue="sex",
    fill=True,
)


plt.xlabel("Flipper Length (mm)")
plt.ylabel("Bill Depth (mm)")
plt.title('Question 25')
plt.show()

# Calculate correlation coefficient for each sex group
correlation_male = df[df['sex'] == 'Male'][['flipper_length_mm', 'bill_depth_mm']].corr().iloc[0, 1]
correlation_female = df[df['sex'] == 'Female'][['flipper_length_mm', 'bill_depth_mm']].corr().iloc[0, 1]

print(f"Correlation coefficient for males: {correlation_male}")
print(f"Correlation coefficient for females: {correlation_female}")


#%% - Q26

sns.set_style("darkgrid")
fig, axes = plt.subplots(1, 3, figsize=(8, 4))
sns.kdeplot(
    data=df,
    x="bill_length_mm",
    y="bill_depth_mm",
    hue="sex",
    fill=True,
    common_norm=False,
    ax=axes[0]
)
axes[0].set_xlabel("Bill Length (mm)")
axes[0].set_ylabel("Bill Depth (mm)")


# Bivariate distribution: bill_length_mm vs flipper_length_mm for male and female
sns.kdeplot(
    data=df,
    x="bill_length_mm",
    y="flipper_length_mm",
    hue="sex",
    fill=True,
    common_norm=False,
    ax=axes[1]
)
axes[1].set_xlabel("Bill Length (mm)")
axes[1].set_ylabel("Flipper Length (mm)")


# Bivariate distribution: flipper_length_mm vs bill_depth_mm for male and female
sns.kdeplot(
    data=df,
    x="flipper_length_mm",
    y="bill_depth_mm",
    hue="sex",
    fill=True,
    common_norm=False,
    ax=axes[2]
)
axes[2].set_xlabel("Flipper Length (mm)")
axes[2].set_ylabel("Bill Depth (mm)")


plt.suptitle('Question 26')
plt.tight_layout()

plt.show()

#%% - Q27

sns.displot(df, x="bill_length_mm", y="bill_depth_mm", hue = 'sex')

plt.title("Question 27")
plt.tight_layout()
plt.show()

#%% - Q28

sns.displot(df, x="bill_length_mm", y="flipper_length_mm", hue = 'sex')

plt.title("Question 28")
plt.tight_layout()
plt.show()

#%% - Q29

sns.displot(df, y="bill_length_mm", x="flipper_length_mm", hue = 'sex')

plt.title("Question 29")
plt.tight_layout()
plt.show()

