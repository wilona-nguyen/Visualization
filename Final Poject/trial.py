
#%%
import plotly.graph_objects as go

# Sample hierarchical data
data = {
    'labels': ['Parent', 'Child 1', 'Child 2', 'Grandchild 1', 'Grandchild 2'],
    'parents': ['', 'Parent', 'Parent', 'Child 1', 'Child 1'],
    'values': [10, 20, 30, 40, 50]
}

# Create Sunburst chart
fig = go.Figure(go.Sunburst(
    labels=data['labels'],
    parents=data['parents'],
    values=data['values']
))

# Show the figure
fig.show()

