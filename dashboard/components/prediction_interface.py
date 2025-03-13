import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import requests
import json
import os
from utils.data_loader import colors

# Get API URL from environment or use default
API_URL = os.environ.get('API_URL', 'http://localhost:8000')

def get_prediction_interface_layout(df):
    """
    Creates a layout for the prediction interface component using actual data columns
    """
    # Get unique values for dropdown options from the specified columns
    column_options = {
        'parte_cuerpo': df['id_parte_cuerpo_igatepmafurat'].unique() if 'id_parte_cuerpo_igatepmafurat' in df.columns else [],
        'municipio': df['id_municipio_at_igatepmafurat'].unique() if 'id_municipio_at_igatepmafurat' in df.columns else [],
        'tipo_jornada': df['ind_tipo_jornada_at_igatepmafurat'].unique() if 'ind_tipo_jornada_at_igatepmafurat' in df.columns else [],
        'realizando_trabajo': df['ind_realizando_trabajo_hab_at_igatepmafurat'].unique() if 'ind_realizando_trabajo_hab_at_igatepmafurat' in df.columns else [],
    }
    
    # Convert numpy arrays to lists and sort for dropdown display
    for key in column_options:
        column_options[key] = sorted([str(x) for x in column_options[key] if str(x) != 'nan'])
    
    # Define column labels and descriptions
    column_labels = {
        'parte_cuerpo': 'Parte del Cuerpo Afectada',
        'municipio': 'Municipio',
        'tipo_jornada': 'Tipo de Jornada',
        'realizando_trabajo': 'Realizando Trabajo Habitual'
    }
    
    column_descriptions = {
        'parte_cuerpo': 'Parte del cuerpo que resultó afectada en el accidente',
        'municipio': 'Municipio donde ocurrió el accidente',
        'tipo_jornada': 'Tipo de jornada laboral durante el accidente',
        'realizando_trabajo': 'Indica si el trabajador realizaba su trabajo habitual'
    }
    
    # Define a muted text color
    muted_text_color = '#cccccc'  # Lighter version of white text
    
    return html.Div([
        html.H3('Predicción de Accidentes', style={'margin-bottom': '20px', 'color': colors['text']}),
        
        html.Div([
            # First row of inputs
            html.Div([
                html.Div([
                    html.Label(column_labels['parte_cuerpo'] + ':', 
                              style={'fontWeight': 'bold', 'margin-bottom': '5px', 'color': colors['text']}),
                    html.Div([
                        dcc.Dropdown(
                            id='parte-cuerpo-dropdown',
                            options=[{'label': i, 'value': i} for i in column_options['parte_cuerpo']],
                            placeholder=f'Seleccione {column_labels["parte_cuerpo"].lower()}',
                            style={'width': '100%'}
                        ),
                        html.Div(column_descriptions['parte_cuerpo'], 
                                style={'fontSize': '12px', 'color': muted_text_color, 
                                       'margin-top': '5px', 'fontFamily': 'Segoe UI'})
                    ])
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.Label(column_labels['municipio'] + ':', 
                              style={'fontWeight': 'bold', 'margin-bottom': '5px', 'color': colors['text']}),
                    html.Div([
                        dcc.Dropdown(
                            id='municipio-dropdown',
                            options=[{'label': i, 'value': i} for i in column_options['municipio']],
                            placeholder=f'Seleccione {column_labels["municipio"].lower()}',
                            style={'width': '100%'}
                        ),
                        html.Div(column_descriptions['municipio'], 
                                style={'fontSize': '12px', 'color': muted_text_color, 
                                       'margin-top': '5px', 'fontFamily': 'Segoe UI'})
                    ])
                ], style={'width': '48%', 'display': 'inline-block'}),
            ], style={'margin-bottom': '20px', 'display': 'flex'}),
            
            # Second row of inputs
            html.Div([
                html.Div([
                    html.Label(column_labels['tipo_jornada'] + ':', 
                              style={'fontWeight': 'bold', 'margin-bottom': '5px', 'color': colors['text']}),
                    html.Div([
                        dcc.Dropdown(
                            id='tipo-jornada-dropdown',
                            options=[{'label': i, 'value': i} for i in column_options['tipo_jornada']],
                            placeholder=f'Seleccione {column_labels["tipo_jornada"].lower()}',
                            style={'width': '100%'}
                        ),
                        html.Div(column_descriptions['tipo_jornada'], 
                                style={'fontSize': '12px', 'color': muted_text_color, 
                                      'margin-top': '5px', 'fontFamily': 'Segoe UI'})
                    ])
                ], style={'width': '48%', 'display': 'inline-block', 'marginRight': '2%'}),
                
                html.Div([
                    html.Label(column_labels['realizando_trabajo'] + ':', 
                              style={'fontWeight': 'bold', 'margin-bottom': '5px', 'color': colors['text']}),
                    html.Div([
                        dcc.Dropdown(
                            id='realizando-trabajo-dropdown',
                            options=[{'label': i, 'value': i} for i in column_options['realizando_trabajo']],
                            placeholder=f'Seleccione {column_labels["realizando_trabajo"].lower()}',
                            style={'width': '100%'}
                        ),
                        html.Div(column_descriptions['realizando_trabajo'], 
                                style={'fontSize': '12px', 'color': muted_text_color, 
                                       'margin-top': '5px', 'fontFamily': 'Segoe UI'})
                    ])
                ], style={'width': '48%', 'display': 'inline-block'})
            ], style={'margin-bottom': '20px', 'display': 'flex'}),
            
            # Description text area
            html.Div([
                html.Label('Descripción del Accidente:', 
                          style={'fontWeight': 'bold', 'margin-bottom': '5px', 'color': colors['text']}),
                dcc.Textarea(
                    id='descripcion-input',
                    placeholder='Ingrese una descripción detallada del accidente...',
                    style={'width': '100%', 'height': '100px', 'padding': '10px', 
                           'borderRadius': '5px', 'border': f'1px solid {colors["border"]}',
                           'fontFamily': 'Segoe UI'}
                )
            ], style={'margin-bottom': '20px'}),
            
            # Prediction button
            html.Div([
                html.Button('Predecir', 
                           id='predict-button', 
                           style={'backgroundColor': colors['primary'], 
                                 'color': 'white', 
                                 'padding': '10px 20px',
                                 'border': 'none',
                                 'borderRadius': '5px',
                                 'cursor': 'pointer',
                                 'fontFamily': 'Segoe UI',
                                 'fontWeight': 'bold'})
            ], style={'textAlign': 'center', 'margin': '20px 0'}),
            
            # Results display area
            html.Div([
                html.Div(id='prediction-result', 
                        style={'padding': '15px', 
                              'border': f'1px solid {colors["border"]}',
                              'borderRadius': '5px',
                              'backgroundColor': colors['panel'],
                              'minHeight': '50px',
                              'color': colors['text']})
            ])
        ], style={'backgroundColor': colors['background'], 
                 'padding': '20px', 
                 'borderRadius': '10px',
                 'boxShadow': f'0 4px 8px 0 {colors["shadow"]}'})
    ])

def register_prediction_interface_callbacks(app):
    """
    Register callbacks for the prediction interface
    """
    @app.callback(
        Output('prediction-result', 'children'),
        [Input('predict-button', 'n_clicks')],
        [State('parte-cuerpo-dropdown', 'value'),
         State('municipio-dropdown', 'value'),
         State('tipo-jornada-dropdown', 'value'),
         State('realizando-trabajo-dropdown', 'value'),
         State('descripcion-input', 'value')]
    )
    def predict_accident(n_clicks, parte_cuerpo, municipio, tipo_jornada, realizando_trabajo, descripcion):
        if n_clicks is None:
            return html.Div("Seleccione los parámetros y presione 'Predecir' para obtener un resultado", 
                           style={'color': colors['text']})
        
        # Check if all required fields are filled
        if not all([parte_cuerpo, municipio, tipo_jornada, realizando_trabajo]):
            return html.Div("Por favor complete todos los campos obligatorios", 
                           style={'color': colors['secondary'], 'fontWeight': 'bold'})
        
        try:
            # Call the API
            response = requests.post(
                f"{API_URL}/predict",
                json={
                    "parte_cuerpo": parte_cuerpo,
                    "municipio": municipio,
                    "jornada_trabajo": tipo_jornada,
                    "realizando_trabajo": realizando_trabajo,
                    "descripcion": descripcion if descripcion else ""
                },
                timeout=10  # Set a timeout for the request
            )
            
            # Check if request was successful
            if response.status_code == 200:
                result = response.json()
                prediction = result["prediction"]
                
                # Determine color based on prediction
                result_color = colors['secondary'] if prediction == 1 else colors['primary']
                
                return [
                    html.H4("Resultado de la Predicción", 
                            style={'marginTop': '0', 'color': colors['text']}),
                    html.Div([
                        html.Span("Nivel de Triage del Accidente: ", style={'fontWeight': 'bold', 'color': colors['text']}),
                        html.Span(prediction, style={'fontWeight': 'bold', 'color': result_color})
                    ]),
                    html.Hr(style={'margin': '15px 0'}),
                    html.Div([
                        html.P("Parámetros utilizados:", style={'color': colors['text'], 'fontWeight': 'bold'}),
                        html.Ul([
                            html.Li(f"Parte del Cuerpo: {parte_cuerpo}", style={'color': colors['text']}),
                            html.Li(f"Municipio: {municipio}", style={'color': colors['text']}),
                            html.Li(f"Tipo de Jornada: {tipo_jornada}", style={'color': colors['text']}),
                            html.Li(f"Realizando Trabajo: {realizando_trabajo}", style={'color': colors['text']}),
                            html.Li(f"Descripción: {descripcion if descripcion else 'No ingresada'}", 
                                  style={'color': colors['text']})
                        ])
                    ])
                ]
            else:
                return html.Div(f"Error al comunicarse con el API: {response.status_code} - {response.text}", 
                               style={'color': colors['secondary']})
                
        except requests.exceptions.ConnectionError:
            return html.Div("No se pudo conectar con el servidor API. Por favor verifique que esté en funcionamiento.", 
                           style={'color': colors['secondary']})
        except requests.exceptions.Timeout:
            return html.Div("La conexión con el API ha excedido el tiempo de espera. Por favor intente nuevamente.", 
                           style={'color': colors['secondary']})
        except Exception as e:
            return html.Div(f"Error inesperado: {str(e)}", 
                           style={'color': colors['secondary']})