#%% - phase 1: import
import dash as dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd
import plotly.express as px
#%% - phase 2: naming
# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
my_app = dash.Dash('My app',)
                   # external_stylesheets)

#%% - phase 3: making the screen

# my_app.layout = html.Div([html.H1('Select a module in Data Viz'),
#                           dcc.Dropdown(id = 'my_drop',
#                                        options = [
#                                            {'label': 'Introduction', 'value': 'Introduction'},
#                                            {'label': 'Panda', 'value': 'Panda'},
#                                            {'label': 'Seaborn', 'value': 'Seaborn'},
#                                            {'label': 'PCA', 'value': 'PCA'},
#                                            {'label': 'Outlier Detection', 'value': 'Outlier Detection'},
#                                        ],
#                                        value = 'Panda',
#                                        multi = True,),
#                           #Output Section
#                           html.Div(id='my_out'),
#
#
#                           ]
#
# )

# my_app.layout = html.Div([html.H1('Hello World! with H1.html', style={'textAlign': 'center', 'color': 'blue'}),
#                           html.H2('Hello World! with H2.html', style={'textAlign': 'center', 'color': 'green'}),
#                           html.H3('Hello World! with H3.html', style={'textAlign': 'center', 'color': 'red'}),
#                           html.H4('Hello World! with H4.html', style={'textAlign': 'center', 'color': 'orange'}),
#                           html.H5('Hello World! with H5.html', style={'textAlign': 'center', 'color': 'black'}),
#                           html.H6('Hello World! with H6.html', style={'textAlign': 'center', 'color': 'yellow'}),
#
# ])

# @my_app.callback(
#     Output(component_id='my_out', component_property='children'),
#     [Input(component_id='my_drop', component_property='value')],
# )
#
# def update_w(x1):
#     return f'The selected module is {x1}'
# ---------------------------------------------------------
my_app.layout = html.Div([
    dcc.Slider(
        id = 'slider_in',
        min = 0,
        max = 20,
        step = 1,
        value = 10
    ),

    html.Div(id='slider_out')
])

@my_app.callback(
    Output(component_id='slider_out', component_property='children'),
    [Input(component_id='slider_in', component_property='value')],
)

def update_w(x1):
    return f'You have selected {x1}'


my_app.run(
    port = 8051,
    host = '0.0.0.0',

)

# my_app.server.run(debug=False)

#%%
# Write a python code that converts F temp to C temp
# two slider: slider 1 input, slider 2 output
# equation: (F-32)/1.8 = C
# title: slider 1 F, slider 2 corresponding C
my_app.layout = html.Div([html.H3('Select a Temperature in F'),
    dcc.Slider(
        id = 'slider1',
        min = 0,
        max = 90,
        step = 5,
        value = 70,
    ),

    html.Br(),
    html.Br(),

    html.H3('Converted Temperature in C'),

    dcc.Slider(
        id = 'slider2',
        min = -10,
        max = 35,
        step = 1
    ),



])

@my_app.callback(
    Output(component_id='slider2', component_property='value'),
    [Input(component_id='slider1', component_property='value')],
)

def update_temp(x1):
    return f' {(x1 - 32)/1.8}'

my_app.run(
    port = 8051,
    host = '0.0.0.0',

)

#%%
my_app.layout = html.Div([
    dcc.Input(id = 'my-input', type = 'number', value = ''),
    html.Div([dcc.Graph(id = 'fig1',)])
])

@my_app.callback(
    Output(component_id='fig1', component_property='figure'),
    [Input(component_id='my-input', component_property='value')],
)

def update_temp(f):
    x = np.linspace(0, 2*np.pi, 1000)
    y = np.sin(2*np.pi*f*x)

    df = pd.DataFrame(dict(
        x=x,
        y=y
    ))

    fig1 = px.line(df, x='x', y='y',
                   width = 1000, height = 800,)

    return fig1

my_app.run(
    port = 8051,
    host = '0.0.0.0',

)