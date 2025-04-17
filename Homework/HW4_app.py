import dash
import numpy as np
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px

#%%
external_sheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

my_app = dash.Dash('Homework 4', external_stylesheets=external_sheets)

my_app.layout = html.Div([html.H1('Homework 4', style={'textAlign': 'center'}),

                          html.Br(),

                          dcc.Tabs(
                              id = 'hw-questions',
                              children=[
                                  dcc.Tab(label='Question 1', value='q1'),
                                  dcc.Tab(label='Question 2', value='q2'),
                                  dcc.Tab(label='Question 3', value='q3'),
                                  dcc.Tab(label='Question 4', value='q4'),
                                  dcc.Tab(label='Question 5', value='q5'),
                                  dcc.Tab(label='Question 6', value='q6'),
                              ]
                          ),
    html.Div(id = 'layout')
])

#====================================
# Layout for Question 1
#====================================


