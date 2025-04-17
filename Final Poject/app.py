import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

# Load data
url = "https://raw.githubusercontent.com/wilona-nguyen/Visualization/refs/heads/main/Final%20Poject/thyroid_cancer_risk_data.csv"
df = pd.read_csv(url)

# Initialize Dash app
app = dash.Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("Thyroid Cancer Risk Dashboard", style={'textAlign': 'center'}),

    dcc.Tabs(
        id="country-tabs",
        value=df["Country"].dropna().unique()[0],  # default country
        children=[
            dcc.Tab(label=country, value=country)
            for country in df["Country"].dropna().unique()
        ]
    ),

    html.Div(id='country-tab-content')
])

#===========================Callback==========================
@app.callback(
    Output('country-tab-content', 'children'),
    Input('country-tabs', 'value')
)
def render_country_tab(country):
    dff = df[df["Country"] == country]

    return html.Div([
        html.H3(f"Visualizations for {country}"),

        dcc.Graph(
            figure=px.box(dff, x='Gender', y='TSH_Level', color='Gender',
                          title=f"TSH Level by Gender in {country}")
        ),
        dcc.Graph(
            figure=px.violin(dff, x='Gender', y='T3_Level', color='Gender', box=True, points='all',
                             title=f"T3 Level by Gender in {country}")
        ),
        dcc.Graph(
            figure=px.strip(dff, x='Gender', y='T4_Level', color='Gender',
                            title=f"T4 Level by Gender in {country}")
        ),
        dcc.Graph(
            figure=px.box(dff, x='Gender', y='Nodule_Size', color='Gender',
                          title=f"Nodule Size by Gender in {country}")
        ),
        dcc.Graph(
            figure=px.bar(
                dff.groupby(['Thyroid_Cancer_Risk', 'Gender']).size().reset_index(name='count'),
                x='Thyroid_Cancer_Risk', y='count', color='Gender', barmode='group',
                title=f"Cancer Risk Level Distribution in {country}"
            )
        ),
        dcc.Graph(
            figure=px.pie(
                dff, names='Diagnosis', title=f"Diagnosis Breakdown in {country}"
            )
        )
    ])



if __name__ == '__main__':
    app.run(debug=True)
