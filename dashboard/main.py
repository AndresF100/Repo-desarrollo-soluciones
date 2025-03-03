import dash
import dash_core_components as dcc
import dash_html_components as html

# Import our custom modules
from utils.data_loader import load_data, generate_and_encode_wordcloud, colors
from components.data_exploration import get_exploration_layout, register_exploration_callbacks
from components.advanced_metrics import get_advanced_metrics_layout, register_advanced_metrics_callbacks
from components.confusion_matrix import get_confusion_matrix_layout, register_confusion_matrix_callbacks

# Initialize the Dash app
app = dash.Dash(__name__)

# Load data
df, predictions_df = load_data()
encoded_image = generate_and_encode_wordcloud(df)

# Define the overall app layout
app.layout = html.Div([
    # Header
    html.H1('Proyecto de Desarrollo de Soluciones', 
            style={'textAlign': 'center', 'fontFamily': 'Segoe UI', 'color': colors['text']}),
    html.H2('Análisis de Accidentes Laborales', 
            style={'textAlign': 'center', 'fontFamily': 'Segoe UI', 'color': colors['text']}),
    
    # Tab system for different sections
    dcc.Tabs([
        dcc.Tab(label='Exploración de Datos', children=[
            get_exploration_layout(df, encoded_image)
        ], style={'backgroundColor': colors['panel'], 'color': colors['text']}),
        
        dcc.Tab(label='Matriz de Confusión', children=[
            get_confusion_matrix_layout(predictions_df)
        ], style={'backgroundColor': colors['panel'], 'color': colors['text']}),
        
        dcc.Tab(label='Métricas Avanzadas', children=[
            get_advanced_metrics_layout(predictions_df)
        ], style={'backgroundColor': colors['panel'], 'color': colors['text']})
    ], style={'fontFamily': 'Segoe UI', 'margin': '20px 0'})
    
], style={'backgroundColor': colors['background'], 'padding': '20px', 'fontFamily': 'Segoe UI'})

# Register all callbacks
register_exploration_callbacks(app, df)
register_confusion_matrix_callbacks(app, predictions_df)
register_advanced_metrics_callbacks(app, predictions_df)

# Run the app
if __name__ == '__main__':
    print("Starting the Dash application...")
    app.run_server(debug=True, port=8050)