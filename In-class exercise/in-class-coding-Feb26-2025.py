#%%
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

iris = px.data.iris()
tip = px.data.tips()
flight = sns.load_dataset('flights')

#%%
# ===================
# Simple Line plot
# ===================
x = np.linspace(-4,4, 100)
y = x ** 2
z = x ** 3

df = pd.DataFrame(np.vstack((x,y,z)).T, columns=['x','y','z'])

fig = px.line(data_frame=df,
              x = 'x',
              y = ['y','z'],
              template='seaborn',
              title=r'\textit {plot} of $x^2, x^3$',
              width=800,
              height=400)
fig.update_layout(
    title_font_family = 'Times New Roman',
    title_font_size = 20,
    title_font_color = 'red',
    legend_title_font_color = 'green',
    legend_title_font_size = 20,
    font_family = 'Courier New',
    font_size = 20,
    font_color = 'blue',
    title_x = 0.5,
    title_y = 0.9,
    margin = dict(l=50, r=20,t = 50, b=20),
    # hovermode = True,
)

fig.update_traces(
    line = dict(width = 4)
)
fig.show(renderer = 'browser')
#%%
df = px.data.gapminder().query("year == 2007").sort_values('pop')

fig = px.bar(df,
             x = 'continent',
             y = 'pop',
             hover_data=['continent','pop','country'])
fig.update_layout(
    xaxis = {"categoryorder":"total ascending"}
)

#%%
df = px.data.tips()
fig = px.bar(df.sort_values('total_bill'),
             x = 'total_bill',
             y = "day",
             color='sex',
             barmode='group')
fig.update_layout(
    yaxis = {"categoryorder":"total ascending"}
)

fig.show(renderer = 'browser')

#%%
fig = px.histogram(df,
                   x = 'total_bill',
                   color='smoker',
                   nbins=50)
fig.show(renderer = 'browser')

#%%
# =============
# Violin Plot
# ===================
fig = px.violin(tip, x="day",
                y="total_bill")
fig.show(renderer="browser")
#%%
# =============
# Pie Plot
# ===================
import plotly.graph_objects as go
labels = ['Oxygen','Hydrogen','Carbon_Dioxide','Nitrogen']
values = [4500, 2500, 1053, 500]

fig = go.Figure(data = [go.Pie(labels=labels,
                               values=values,
                               pull=[0.01, 0.01, 0.05, 0.01])])
fig.show(renderer='browser')

#%% Animation using plotly
df = px.data.gapminder()
fig = px.scatter(df,
                 x = 'gdpPercap',
                 y = 'lifeExp',
                 size= 'pop',
                 color='continent',
                 size_max=60,
                 hover_name='country',
                title='GDP per capital per continent',
                animation_frame='year',
                animation_group='country',
                range_x=[300,60000],
                range_y=[25,90]
                 )
fig.show(renderer='browser')
#%% ===============================
# choropleth
# ============================
fig = px.choropleth(df,
                    locations='iso_alpha',
                    title = 'life Expectancy',
                    projection='natural earth',
                    width=800, height=800,
                    color = 'lifeExp',
                    hover_name='country',
                    animation_frame='year',
                    color_continuous_scale=px.colors.sequential.Plasma)
fig.show(renderer='browser')

#%%
# ==================
# Regression Line
# ===================
df = px.data.tips()
fig = px.scatter(df,
                 x ='total_bill',
                 y='tip',
                 trendline='ols',
                 trendline_color_override='red',

)
fig.show(renderer='browser')
#%%===========================
# 3D plot and contour plot
# ============================
x = np.arange(-1,1,0.1)
y = np.arange(-1,1,0.1)
X,Y = np.meshgrid(x,y)
Z = (Y-X)**4 + 8* X * Y - X * Y + 3
surface = go.Surface(
            x= X,
            y = Y,
            z = Z
)
fig = go.Figure(surface)
fig.show(renderer='browser')

#%% contour plot

fig = go.Figure(data=go.Contour(x = x,y=y,z=Z,
                                colorscale='rainbow',
                                contours=dict(
                                start=-1,
                                end=5,
                                size=.1,
                                # coloring = 'lines',
                                showlabels = True)))
fig['layout'].update(
    height = 800, width =800,
    title_text="Non-Quadratic Contour Plot",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Local Min', x= -0.5, y=0.5, font_size=20, showarrow=False),
                 dict(text='Global Min', x=0.5, y=-0.5, font_size=20, showarrow=False)])

fig.show(renderer='browser')


