import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd

import plotly.express as px

# Generate random data for scatter plot
np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})

# Create a scatter plot
fig = px.scatter(df, x='x', y='y', title='Random Scatter Plot')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div(children=[
    html.H1(children='Proyecto Desarrollo de Soluciones: Grupo 2'),
    dcc.Graph(
        id='scatter-plot',
        figure=fig
    )
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)