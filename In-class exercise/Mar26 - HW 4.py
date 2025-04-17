#%%
import dash
import numpy as np
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px

#%%
external_sheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#%%
my_app = dash.Dash('Homework 4', external_stylesheets=external_sheets)

my_app.layout = html.Div([dcc.Graph(id = 'my-graph'),

                    html.P('Mean'),
                    dcc.Slider(id = 'mean',
                               min = -3,
                               max = 3,
                               value = 0,
                               marks = {'-3': '-3',
                                        '-2': '-2',
                                        '-1': '-1',
                                        '0': '0',
                                        '1': '1',
                                        '2': '2',
                                        '3': '3',
                                        }),

                    html.Br(),

                    html.P('Std'),
                    dcc.Slider(id='std',
                                min=1,
                                max=3,
                                value=1,
                                marks={ '1': '1',
                                        '2': '2',
                                        '3': '3',
                                            }),

                    html.Br(),

                    html.P('Number of Observations'),
                    dcc.Slider(id='size',
                                min=1,
                                max=10000,
                                value=100,
                                marks={ 100: '100',
                                        500: '500',
                                        1000: '1000',
                                        10000: '10000',
                                            }),

                    html.Br(),

                    dcc.Dropdown(id='bins',
                                 options = [
                                     {'label': 20, 'value': 20},
                                     {'label': 40, 'value': 40},
                                     {'label': 60, 'value': 60},
                                     {'label': 80, 'value': 80},
                                     {'label': 100, 'value': 100},
                                 ], value = 20,clearable = False)


], style = {'width': '75%', 'height': '100px'})

@my_app.callback(
    Output(component_id = 'my-graph', component_property = 'figure'),
    [Input(component_id = 'mean', component_property = 'value'),
     Input(component_id = 'std', component_property = 'value'),
     Input(component_id = 'size', component_property = 'value'),
     Input(component_id = 'bins', component_property = 'value')],
)

def w(mean, std, size, bins):
    x = np.random.normal(loc=mean, scale=std, size=size)
    fig = px.histogram(x =  x,
                       nbins = bins,
                       range_x = [-5,5],
                       width = 1600, height = 800,)
    return fig

my_app.run(
    port = 8082,
    host = '0.0.0.0'
)