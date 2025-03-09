import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import pandas as pd
from dash.dependencies import Input, Output
import plotly.express as px
from analisis_texto import generate_wordcloud
import base64

# Load the data
df = pd.read_csv('c:/Users/Santiago Calderón/OneDrive - LLA/Documents/Maestria/Proyecto Final/Proyecto de Materia/Taller 1/Repo-desarrollo-soluciones/dash/data_for_visuals.csv.xls')

# Prepare the day column
df['day'] = pd.to_datetime(df['fecha_siniestro_igdacmlmasolicitudes']).dt.day

# Generate wordcloud image file
wordcloud_path = "c:/Users/Santiago Calderón/OneDrive - LLA/Documents/Maestria/Proyecto Final/Proyecto de Materia/Taller 1/Repo-desarrollo-soluciones/dash/wordcloud.png"
wordcloud_result = generate_wordcloud(df, output_path=wordcloud_path, show_plot=False)

# Encode the wordcloud image
encoded_image = base64.b64encode(open(wordcloud_path, 'rb').read()).decode('ascii')

# Initialize the Dash app
app = dash.Dash(__name__)

# Define colors
colors = {
    'background': '#2c3e50',  # Grayish blue
    'text': '#FFFFFF',        # White text for better contrast
    'panel': '#34495e'        # Slightly lighter grayish blue for panels
}

# Define the layout with a grid structure
app.layout = html.Div([

    html.H1('Proyecto de Desarrollo de Soluciones', 
            style={'textAlign': 'center', 'fontFamily': 'Segoe UI', 'color': colors['text']}),
    html.H2('Análisis de Accidentes Laborales', 
            style={'textAlign': 'center', 'fontFamily': 'Segoe UI', 'color': colors['text']}),

    # Filters section at the top
    html.Div([
        html.Div([
            html.H3('Municipio', style={'marginRight': '2%', 'fontFamily': 'Segoe UI', 'color': colors['text']}),
            dcc.Dropdown(
                id='municipio_dropdown',
                options=[{'label': 'Todos', 'value': 'all'}] + 
                        [{'label': str(i), 'value': i} for i in sorted(df['id_municipio_at_igatepmafurat'].unique())],
                value='all',
                clearable=False
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
        
        html.Div([
            html.H3('Tipo de Accidente', style={'marginRight': '2%', 'fontFamily': 'Segoe UI', 'color': colors['text']}),
            dcc.Dropdown(
                id='origen_dropdown',
                options=[{'label': 'Todos', 'value': 'all'}] + 
                        [{'label': str(i), 'value': i} for i in sorted(df['origen_igdactmlmacalificacionorigen'].unique())],
                value='all',
                clearable=False
            ),
        ], style={'width': '48%', 'display': 'inline-block'}),
    ], style={'padding': '20px', 'backgroundColor': colors['panel'], 'borderRadius': '10px', 'margin': '10px'}),
    
    # Grid layout for charts (2x2)
    html.Div([
        # Row 1
        html.Div([
            # Column 1
            html.Div([
                html.H3('Accidentes por hora del día', 
                        style={'marginRight': '2%', 'fontFamily': 'Segoe UI', 'color': colors['text']}),
                dcc.Graph(id='hour_graph'),
            ], style={'width': '46%', 'display': 'inline-block', 'marginRight': '2%', 
                     'backgroundColor': colors['panel'], 'padding': '10px', 'borderRadius': '10px'}),
            
            # Column 2
            html.Div([
                html.H3('Accidentes por día del mes', 
                        style={'marginRight': '2%', 'fontFamily': 'Segoe UI', 'color': colors['text']}),
                dcc.Graph(id='day_graph'),
            ], style={'width': '46%', 'display': 'inline-block', 
                     'backgroundColor': colors['panel'], 'padding': '10px', 'borderRadius': '10px'}),
        ], style={'marginBottom': '20px'}),
        
        # Row 2
        html.Div([
            # Column 1
            html.Div([
                html.H3('Accidentes por Municipio', 
                        style={'marginRight': '2%', 'fontFamily': 'Segoe UI', 'color': colors['text']}),
                dcc.Graph(id='municipio_graph'),
            ], style={'width': '46%', 'display': 'inline-block', 'marginRight': '2%', 
                     'backgroundColor': colors['panel'], 'padding': '10px', 'borderRadius': '10px'}),
            
            # Column 2
            html.Div([
                html.H3('Accidentes por Tipo (Balance de clases)', 
                        style={'marginRight': '2%', 'fontFamily': 'Segoe UI', 'color': colors['text']}),
                dcc.Graph(id='origen_graph'),
            ], style={'width': '46%', 'display': 'inline-block', 
                     'backgroundColor': colors['panel'], 'padding': '10px', 'borderRadius': '10px'}),
        ], style={'marginBottom': '20px'}),
        
        # Row 3 - WordCloud (full width)
        html.Div([
            html.H3('Análisis de Texto - Palabras más frecuentes en descripciones de accidentes',
                    style={'textAlign': 'center', 'fontFamily': 'Segoe UI', 'color': colors['text']}),
            html.Img(src='data:image/png;base64,{}'.format(encoded_image), 
                    style={'width': '92%', 'height': 'auto', 'display': 'block', 'margin': '0 auto'}),
        ], style={'backgroundColor': colors['panel'], 'padding': '20px', 'borderRadius': '10px', 'width': '90%', 'margin': '0 auto'})
        
    ], style={'padding': '10px'})
], style={'backgroundColor': colors['background'], 'padding': '20px', 'fontFamily': 'Segoe UI'})

# Define callbacks for all graphs to respond to both filters
@app.callback(
    [Output('hour_graph', 'figure'),
     Output('day_graph', 'figure'),
     Output('municipio_graph', 'figure'),
     Output('origen_graph', 'figure')],
    [Input('municipio_dropdown', 'value'),
     Input('origen_dropdown', 'value')]
)
def update_graphs(selected_municipio, selected_origen):
    # Filter data based on selections
    filtered_df = df.copy()
    
    if selected_municipio != 'all':
        filtered_df = filtered_df[filtered_df['id_municipio_at_igatepmafurat'] == selected_municipio]
    
    if selected_origen != 'all':
        filtered_df = filtered_df[filtered_df['origen_igdactmlmacalificacionorigen'] == selected_origen]
    
    # Define common figure layout settings
    layout_settings = {
        'paper_bgcolor': colors['panel'],
        'plot_bgcolor': colors['panel'],
        'font': {'color': colors['text'], 'family': 'Segoe UI'},
        'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40}
    }
    
    # Create the four figures
    fig_hour = px.histogram(filtered_df, x='hora_at_igatepmafurat', 
                           title='Accidentes por hora del día')
    fig_hour.update_layout(**layout_settings)
    
    fig_day = px.histogram(filtered_df, x='day', 
                          title='Accidentes por día del mes')
    fig_day.update_layout(**layout_settings)
    
    fig_mun = px.histogram(filtered_df, x='id_municipio_at_igatepmafurat', 
                          title='Accidentes por Municipio')
    fig_mun.update_layout(**layout_settings)
    
    fig_origen = px.histogram(filtered_df, x='origen_igdactmlmacalificacionorigen', 
                             title='Accidentes por Tipo (Balance de clases)')
    fig_origen.update_layout(**layout_settings)
    
    return fig_hour, fig_day, fig_mun, fig_origen

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=8050)