#%%
import seaborn as sns
import pandas as pd
import dash
import numpy as np
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import matplotlib.pyplot as plt

#%%

tips = sns.load_dataset('tips')

#%%
external_sheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
my_app = dash.Dash('Quiz', external_stylesheets=external_sheets)

#%% -a
a_layout = html.Div([
    html.Label('Please select the feature from the menu'),
    dcc.Dropdown(id='drop_a',
                 options=[
                     {'label': 'Day', 'value': 'Day'},
                     {'label': 'Time', 'value': 'Time'},
                     {'label': 'Sex', 'value': 'Sex'},
                 ], value = 'Day'),
    html.Br(),

    html.Label('Please select the output variable to be plotted'),
    dcc.Dropdown(id='drop_b',
                 options=[
                     {'label': 'total_bill', 'value': 'total_bill'},
                      {'label': 'tip', 'value': 'tip'},
                 ], value = 'total_bill'),
    html.Br(),

    dcc.Graph(id = 'output-graph'),

])

@my_app.callback(
    Output('output-graph', 'figure'),
    [Input('drop_a', 'value'),
    Input('drop_b', 'value')],
)

def pie_plot(drop_a, drop_b):
    fig = px.pie(tips, values=drop_b, names=drop_a)

    return fig

my_app.run(
    port = 8082,
    host = '0.0.0.0'
)