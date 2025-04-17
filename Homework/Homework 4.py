#%%
import math
import pandas as pd
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FixedLocator, FixedFormatter

import dash
import numpy as np
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go


#%%
external_sheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
my_app = dash.Dash('Homework 4', external_stylesheets=external_sheets)

#====================================
# Main Layout
#====================================

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



#%% - Question 1
url = 'https://raw.githubusercontent.com/rjafari979/Information-Visualization-Data-Analytics-Dataset-/refs/heads/main/CONVENIENT_global_confirmed_cases.csv'

df = pd.read_csv(url)
df = df.dropna()

df['China_sum'] = df.filter(like='China.').apply(pd.to_numeric, errors='coerce').sum(axis=1)
df['United Kingdom_sum'] = df.filter(like='United Kingdom.').apply(pd.to_numeric, errors='coerce').sum(axis=1)
df['date'] = pd.to_datetime(df['Country/Region'], format='%m/%d/%y')
countries = ['US', 'Brazil', 'United Kingdom_sum', 'China_sum', 'India', 'Italy', 'Germany']

df_plot = df[['date'] + countries]


#====================================
# Layout for Question 1
#====================================

q1_layout = html.Div([
    html.H1("COVID-19 Global Confirmed Cases Over Time", style={'textAlign': 'center'}),

    html.Label("Pick the country Name", style={'fontWeight': 'bold', 'marginTop': '20px'}),
    dcc.Dropdown(
        id='country-dropdown',
        options=[{'label': country, 'value': country} for country in countries],
        value=['US'],
        multi=True
    ),

    dcc.Graph(id='covid-graph'),

])

#====================================
# Callbacks for Question 1
#====================================
@my_app.callback(
    Output(component_id='covid-graph', component_property='figure'),
    Input(component_id='country-dropdown', component_property='value')
)

def update_q1(selected_countries):
    traces = []
    for country in selected_countries:
        traces.append(go.Scatter(
            x=df_plot['date'],
            y=df_plot[country],
            mode='lines',
            name=country
        ))

    return {
        'data': traces,
        'layout': go.Layout(
            title='Confirmed COVID-19 Cases by Country',
            xaxis={'title': 'Date'},
            yaxis={'title': 'Confirmed Cases'},
            hovermode='closest'
        )
    }

#====================================
# Layout for Question 2
#====================================

q2_layout = html.Div([
    dcc.Markdown("### Quadratic Function Plot:  \n$$f(x) = ax^2 + bx + c$$", mathjax=True),

    html.Div([
        html.Label("Select coefficient a:"),
        dcc.Slider(id='a-slider', min=-10, max=10, step=0.5, value=1,
                   marks={i: str(i) for i in range(-10, 11)}),
        html.Br(),

        html.Label("Select coefficient b:"),
        dcc.Slider(id='b-slider', min=-10, max=10, step=0.5, value=0,
                   marks={i: str(i) for i in range(-10, 11)}),
        html.Br(),

        html.Label("Select coefficient c:"),
        dcc.Slider(id='c-slider', min=-10, max=10, step=0.5, value=0,
                   marks={i: str(i) for i in range(-10, 11)})
    ], style={'margin': '30px'}),

    dcc.Graph(id='quadratic-graph')
])


#====================================
# Callbacks for Question 2
#====================================

@my_app.callback(
    Output('quadratic-graph', 'figure'),
    Input('a-slider', 'value'),
    Input('b-slider', 'value'),
    Input('c-slider', 'value')
)
def update_quadratic(a, b, c):
    x = np.linspace(-2, 2, 1000)
    y = a * x**2 + b * x + c

    trace = go.Scatter(x=x, y=y, mode='lines', name=f'f(x) = {a}x² + {b}x + {c}')
    return {
        'data': [trace],
        'layout': go.Layout(
            title='Plot of f(x) = ax² + bx + c',
            xaxis={'title': 'x'},
            yaxis={'title': 'f(x)'},
            hovermode='closest'
        )
    }

#====================================
# Layout for Question 3
#====================================

q3_layout = html.Div([
    html.H3("Dash Calculator App", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Enter value for a:"),
        dcc.Input(id='input-a', type='number', value=1, step=0.1),
        html.Br(), html.Br(),

        html.Label("Enter value for b:"),
        dcc.Input(id='input-b', type='number', value=1, step=0.1),
        html.Br(), html.Br(),

        html.Label("Choose operation:"),
        dcc.Dropdown(
            id='operation-dropdown',
            options=[
                {'label': 'Addition (a + b)', 'value': 'add'},
                {'label': 'Subtraction (a - b)', 'value': 'sub'},
                {'label': 'Multiplication (a * b)', 'value': 'mul'},
                {'label': 'Division (a / b)', 'value': 'div'},
                {'label': 'Logarithm (log_b(a))', 'value': 'log'},
                {'label': 'b-th Root of a', 'value': 'root'}
            ],
            value='add'
        )
    ], style={'margin': '30px'}),

    html.Div(id='calculator-output', style={'fontWeight': 'bold', 'fontSize': '24px', 'textAlign': 'center'})
])

#====================================
# Callbacks for Question 3
#====================================

@my_app.callback(
    Output('calculator-output', 'children'),
    Input('input-a', 'value'),
    Input('input-b', 'value'),
    Input('operation-dropdown', 'value')
)
def calculate(a, b, operation):
    try:
        if operation == 'add':
            result = a + b
            return f"{a} + {b} = {result}"

        elif operation == 'sub':
            result = a - b
            return f"{a} - {b} = {result}"

        elif operation == 'mul':
            result = a * b
            return f"{a} * {b} = {result}"

        elif operation == 'div':
            if b == 0:
                return "Error: Division by zero is not allowed."
            result = a / b
            return f"{a} / {b} = {result}"

        elif operation == 'log':
            if a <= 0:
                return "Error: 'a' must be positive for logarithm."
            if b <= 1:
                return "Error: 'b' must be > 1 for logarithm base."
            result = math.log(a, b)
            return f"log base {b} of {a} = {result}"

        elif operation == 'root':
            if a == 0:
                return "Error: 'a' cannot be zero for root."
            if b <= 0 or not float(b).is_integer():
                return "Error: 'b' must be a positive integer."
            if a < 0 and int(b) % 2 == 0:
                return "Error: Even root of negative number is invalid."
            result = a ** (1 / b)
            return f"{b}-th root of {a} = {result}"

    except Exception as e:
        return f"Error: {str(e)}"

#====================================
# Layout for Question 4
#====================================

q4_layout = html.Div([
    html.H3("Polynomial Function Plotter", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Enter the order of the polynomial (n):"),
        dcc.Input(id='poly-order', type='number', value=2, min=0, step=1),
        html.Br(), html.Br(),
    ], style={'margin': '30px'}),

    dcc.Graph(id='polynomial-graph')
])

#====================================
# Callbacks for Question 4
#====================================

@my_app.callback(
    Output('polynomial-graph', 'figure'),
    Input('poly-order', 'value')
)
def update_polynomial_graph(order):
    x = np.linspace(-2, 2, 1000)

    try:
        n = int(order)
        y = x ** n
        label = f"f(x) = x^{n}"
    except:
        y = np.zeros_like(x)
        label = "Invalid input"

    trace = go.Scatter(x=x, y=y, mode='lines', name=label)

    return {
        'data': [trace],
        'layout': go.Layout(
            title=f"Polynomial Plot of x^{order}",
            xaxis={'title': 'x'},
            yaxis={'title': 'f(x)'},
            hovermode='closest'
        )
    }

#====================================
# Layout for Question 5
#====================================

q5_layout = html.Div([
    html.H3("Sinusoidal Signal with White Noise & FFT", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Number of cycles:"),
        dcc.Input(id='cycles-input', type='number', value=3, min=1, step=1),
        html.Br(), html.Br(),

        html.Label("Mean of white noise:"),
        dcc.Input(id='noise-mean-input', type='number', value=0, step=0.1),
        html.Br(), html.Br(),

        html.Label("Standard deviation of white noise:"),
        dcc.Input(id='noise-std-input', type='number', value=0.2, step=0.1),
        html.Br(), html.Br(),

        html.Label("Number of samples:"),
        dcc.Input(id='samples-input', type='number', value=1000, min=10, step=10),
        html.Br(), html.Br(),
    ], style={'margin': '30px'}),

    dcc.Graph(id='signal-plot'),
    dcc.Graph(id='fft-plot')
])

#====================================
# Callbacks for Question 5
#====================================

@my_app.callback(
    [Output('signal-plot', 'figure'),
     Output('fft-plot', 'figure')],
    [Input('cycles-input', 'value'),
     Input('noise-mean-input', 'value'),
     Input('noise-std-input', 'value'),
     Input('samples-input', 'value')]
)
def update_signal_and_fft(cycles, noise_mean, noise_std, samples):
    try:
        samples = int(samples)
        x = np.linspace(-np.pi, np.pi, samples)
        sine_wave = np.sin(cycles * x)
        noise = np.random.normal(noise_mean, noise_std, size=samples)
        signal = sine_wave + noise

        # FFT
        fft_result = fft(signal)
        freqs = np.fft.fftfreq(samples, d=(x[1] - x[0]))

        # Only positive frequencies
        pos_freqs = freqs[:samples // 2]
        pos_fft = np.abs(fft_result[:samples // 2])

        # Plot 1: Signal
        signal_fig = {
            'data': [go.Scatter(x=x, y=signal, mode='lines', name='f(x) = sin(x) + noise')],
            'layout': go.Layout(title='Noisy Sinusoidal Signal',
                                xaxis={'title': 'x'},
                                yaxis={'title': 'f(x)'})
        }

        # Plot 2: FFT
        fft_fig = {
            'data': [go.Scatter(x=pos_freqs, y=pos_fft, mode='lines', name='FFT Magnitude')],
            'layout': go.Layout(title='FFT of the Signal',
                                xaxis={'title': 'Frequency'},
                                yaxis={'title': 'Magnitude'})
        }

        return signal_fig, fft_fig

    except Exception as e:
        return go.Figure(), go.Figure()

#====================================
# Layout for Question 6
#====================================

# Sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Range of input p
p = np.linspace(-5, 5, 1000)

# Layout
q6_layout = html.Div([
    html.H3("Two-Layer Neural Network Output", style={'textAlign': 'center'}),
    dcc.Markdown("$$a^2 = w_{1,1}^2 a_1^1 + w_{1,2}^2 a_2^1 + b_1^2$$", mathjax=True, style={'textAlign': 'center'}),

    dcc.Graph(id='nn-graph'),

    dcc.Markdown("**$$b_1^1$$**", mathjax=True),
    dcc.Slider(id='b11', min=-10, max=10, step=1, value=0),

    dcc.Markdown("**$$b_2^1$$**", mathjax=True),
    dcc.Slider(id='b21', min=-10, max=10, step=1, value=0),

    dcc.Markdown("**$$w_{1,1}^1$$**", mathjax=True),
    dcc.Slider(id='w11_1', min=-10, max=10, step=1, value=1),

    dcc.Markdown("**$$w_{2,1}^1$$**", mathjax=True),
    dcc.Slider(id='w21_1', min=-10, max=10, step=1, value=1),

    dcc.Markdown("**$$b_1^2$$**", mathjax=True),
    dcc.Slider(id='b12', min=-10, max=10, step=1, value=0),

    dcc.Markdown("**$$w_{1,1}^2$$**", mathjax=True),
    dcc.Slider(id='w11_2', min=-10, max=10, step=1, value=1),

    dcc.Markdown("**$$w_{1,2}^2$$**", mathjax=True),
    dcc.Slider(id='w12_2', min=-10, max=10, step=1, value=1),

], style={'width': '30%', 'display': 'inline-block', 'vertical-align':'middle'})

#====================================
# Callbacks for Question 6
#====================================

@my_app.callback(
    Output('nn-graph', 'figure'),
    [Input('b11', 'value'),
     Input('b21', 'value'),
     Input('w11_1', 'value'),
     Input('w21_1', 'value'),
     Input('b12', 'value'),
     Input('w11_2', 'value'),
     Input('w12_2', 'value')]
)
def update_nn(b11, b21, w11_1, w21_1, b12, w11_2, w12_2):
    a1_1 = sigmoid(p * w11_1 + b11)
    a2_1 = sigmoid(p * w21_1 + b21)
    a2 = w11_2 * a1_1 + w12_2 * a2_1 + b12

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=p, y=a2, mode='lines', name='$a^2$',
                             line=dict(color='blue', width=2), opacity=0.7))
    fig.update_layout(
        title='a² vs p',
        xaxis_title='p',
        yaxis_title='a²',
        template='plotly_white'
    )
    return fig



#=======================================
#Final Display
@my_app.callback(
    Output(component_id='layout', component_property='children'),
    Input(component_id='hw-questions', component_property='value'),
)

def update_layout(queues):
    if queues == 'q1':
        return q1_layout
    if queues == 'q2':
        return q2_layout
    if queues == 'q3':
        return q3_layout
    if queues == 'q4':
        return q4_layout
    if queues == 'q5':
        return q5_layout
    elif queues == 'q6':
        return q6_layout

my_app.run(
    port = 8082,
    host = '0.0.0.0'
)