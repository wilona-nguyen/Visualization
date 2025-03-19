#%%
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import seaborn as sns
#%% - 1
df = px.data.stocks()

print("Features:", df.columns.tolist())
print(df.tail())

#%% - 2
fig = go.Figure()

companies = df.columns[1:]
for company in companies:
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df[company],
        mode='lines',
        name=company,
        line=dict(width=4)
    ))


fig.update_layout(
    title="Stock Values - Major Tech Companies",
    title_x=0.5,
    title_font=dict(family="Times New Roman", size=30, color="red"),
    xaxis_title="Time",
    yaxis_title="Normalized ($)",
    xaxis=dict(title_font=dict(family="Courier New", size=30, color="yellow")),
    yaxis=dict(title_font=dict(family="Courier New", size=30, color="yellow")),
    legend=dict(title_font=dict(color="green", size=30)),
    font=dict(family="Courier New", size=30, color="yellow"),
    template="plotly_dark",
    width=2000,
    height=800
)

fig.show(renderer="browser")


#%% - 3
fig = make_subplots(rows=3, cols=2,
                    shared_xaxes=False, shared_yaxes=False)

companies = df.columns[1:]
row, col = 1, 1

for company in companies:
    fig.add_trace(
        go.Histogram(x=df[company], nbinsx=50, name=company),
        row=row, col=col
    )

    # Update row and column indices
    col += 1
    if col > 2:
        col = 1
        row += 1

fig.update_layout(
    title="Histogram Plot",
    title_x=0.5,
    title_font=dict(family="Times New Roman", size=30, color="red"),
    font=dict(family="Courier New"),
    xaxis_title="Normalized Price ($)",
    yaxis_title="Frequency",
    xaxis=dict(title_font=dict(size=15, color="black")),
    yaxis=dict(title_font=dict(size=15, color="black")),
    legend=dict(title_font=dict(color="green", size=30)),
    showlegend=True
)

for row_idx in range(1, 4):
    for col_idx in range(1, 3):
        fig.update_xaxes(title_text="Normalized Price ($)", row=row_idx, col=col_idx)
        fig.update_yaxes(title_text="Frequency", row=row_idx, col=col_idx)

fig.show(renderer="browser")


#%% - 4a
from sklearn.preprocessing import StandardScaler

#%% - 4b
scaler = StandardScaler()
stocks_scaled = scaler.fit_transform(df[companies])

# Perform Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(stocks_scaled, full_matrices=False)

# Singular Values
print("Singular values: \n", S)

#Condition Number
condition_number = S.max() / S.min()
print("Condition Number:", condition_number)

#%% - 4c
corr_mat = df[companies].corr()

# Plot heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(corr_mat, annot=True, fmt=".2f", linewidths=0.5)

# Show the plot
plt.title("Correlation Coefficient between features-Original feature space")
plt.show()

#%% - 4d
from sklearn.decomposition import PCA

pca = PCA()
pca.fit(stocks_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)
n_components = np.argmax(cumulative_variance >= 0.95) + 1

print("Number of features to keep:", n_components)
print("Explained variance ratio (original feature space): \n", explained_variance_ratio)
print("Explained variance ratio (reduced feature space):", cumulative_variance[n_components-1])

features_removed = df[companies].shape[1] - n_components
print("Number of features to be removed:", features_removed)

#%% - 4e
# Plot cumulative explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, marker="o", linestyle="-", color="r")

# Add dashed lines at 95% variance and optimal component count
plt.axhline(y=95, color="red", linestyle="dashed", label="95% Explained Variance")
plt.axvline(x=n_components, color="black", linestyle="dashed", label=f"{n_components} Components")

# Labels and title
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Explained Variance (%)")
plt.title("Cumulative Explained Variance vs Number of Components")
plt.legend()
plt.grid()
plt.show()


#%% - 4f
# Transform the data using the selected number of components
stocks_reduced = pca.transform(stocks_scaled)[:, :n_components]

# Perform Singular Value Decomposition (SVD) on the reduced feature space
U_reduced, S_reduced, Vt_reduced = np.linalg.svd(stocks_reduced, full_matrices=False)

# Compute condition number for the reduced feature space
condition_number_reduced = S_reduced.max() / S_reduced.min()

# Display results
print("Singular values (Original Feature Space): \n", S)
print("Condition Number (Original Feature Space):", condition_number)

print("Singular values (Reduced Feature Space): \n", S_reduced)
print("Condition Number (Reduced Feature Space):", condition_number_reduced)


#%% - 4g
# Convert reduced dataset back to a DataFrame
stocks_reduced_df = pd.DataFrame(stocks_reduced, columns=[f"PC{i+1}" for i in range(n_components)])

# Compute correlation matrix
correlation_matrix_reduced = stocks_reduced_df.corr()

# Plot heatmap using seaborn
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix_reduced, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)

# Show the plot
plt.title("Feature Correlation Heatmap (Reduced Feature Space)")
plt.show()


#%% - 4h
column_names = [f'Principal col {i+1}' for i in range(n_components)]
stocks_pca_df = pd.DataFrame(stocks_reduced, columns=column_names)

print(stocks_pca_df.head())


#%% - 4i


stocks_pca_df['date'] = df['date']

fig = px.line(stocks_pca_df, x='date', y=column_names, title="PCA Transformed Features Over Time")

# Update layout with the specified formatting
fig.update_layout(
    title=dict(
        text="PCA Transformed Features Over Time",
        font=dict(family="Times New Roman", size=30, color="red"),
        x=0.5
    ),
    xaxis=dict(
        title=dict(text="Date", font=dict(family="Courier New", size=30, color="yellow")),
        tickfont=dict(family="Courier New", size=30, color="yellow")
    ),
    yaxis=dict(
        title=dict(text="Transformed Feature Values", font=dict(family="Courier New", size=30, color="yellow")),
        tickfont=dict(family="Courier New", size=30, color="yellow")
    ),
    legend=dict(
        title=dict(font=dict(size=30, color="green")),
        font=dict(family="Courier New", size=30, color="yellow")
    ),
    width=2000,
    height=800,
    template="plotly_dark"
)

# Update traces to set line width
fig.update_traces(line=dict(width=4))


fig.show(renderer="browser")


#%% - 4j
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Number of PCA components
num_components = len(column_names)

# Create a subplot layout with one column and multiple rows
fig = make_subplots(rows=num_components, cols=1,
                    subplot_titles=column_names,
                    vertical_spacing=0.1)

# Add histogram for each principal component
for i, col in enumerate(column_names):
    fig.add_trace(
        go.Histogram(x=stocks_pca_df[col], name=col, marker=dict(color='lightblue')),
        row=i+1, col=1
    )

# Update layout for better visualization
fig.update_layout(
    title=dict(text="Histograms of PCA Transformed Features", font=dict(family="Times New Roman", size=30, color="red"), x=0.5),
    height=300 * num_components,  # Adjust height dynamically based on number of components
    width=1000,
    showlegend=False,
    template="plotly_dark"
)

# Show the plot
fig.show(renderer="browser")

#%% - 4k

# Remove 'Date' column (if it exists) for numerical-only features
stocks_numeric = df.select_dtypes(include=['number'])

# Create scatter matrix for original features
fig_original = px.scatter_matrix(
    stocks_numeric,
    title="Scatter Matrix of Original Feature Space",
    dimensions=stocks_numeric.columns,
)

fig_original.update_traces(diagonal_visible=False)

# Show the plot
fig_original.show(renderer="browser")

#%% - 4k

# Create scatter matrix for PCA-transformed features, with diagonals as histograms
fig_reduced = px.scatter_matrix(
    stocks_pca_df.drop(columns=["date"]),  # Drop Date column if present
    title="Scatter Matrix of Reduced Feature Space",
    dimensions=column_names,
)

fig_reduced.update_traces(diagonal_visible=False)

# Show the plot
fig_reduced.show(renderer="browser")

