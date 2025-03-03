import dash
from dash import dcc  # Changed from import dash_core_components as dcc
from dash import html  # Changed from import dash_html_components as html
import plotly.figure_factory as ff
from dash.dependencies import Input, Output
from sklearn.metrics import confusion_matrix

from utils.data_loader import colors, layout_settings

def get_confusion_matrix_layout(predictions_df):
    """Return the layout for the confusion matrix section"""
    return html.Div([
        html.H3('Matriz de Confusión por Clase',
                style={'textAlign': 'center', 'fontFamily': 'Segoe UI', 'color': colors['text'], 'marginBottom': '20px'}),
        
        # Dropdown for selecting class
        html.Div([
            html.H4('Seleccionar Clase:', style={'fontFamily': 'Segoe UI', 'color': colors['text'], 'marginRight': '20px'}),
            dcc.Dropdown(
                id='class_dropdown',
                options=[{'label': str(i), 'value': i} for i in sorted(predictions_df['y_test'].unique())],
                value=sorted(predictions_df['y_test'].unique())[0],
                clearable=False,
                style={'width': '50%', 'color': '#000000'}
            ),
        ], style={'display': 'flex', 'alignItems': 'center', 'justifyContent': 'center', 'marginBottom': '20px'}),
        
        # Confusion Matrix Graph
        dcc.Graph(id='confusion_matrix'),
        
        # Metrics Display
        html.Div([
            html.Div(id='precision_recall', 
                    style={'textAlign': 'center', 'fontFamily': 'Segoe UI', 'color': colors['text'], 'fontSize': '18px'})
        ], style={'marginTop': '20px'})
        
    ], style={'backgroundColor': colors['panel'], 'padding': '20px', 'borderRadius': '10px', 'width': '90%', 'margin': '20px auto'})

def register_confusion_matrix_callbacks(app, predictions_df):
    @app.callback(
        [Output('confusion_matrix', 'figure'),
         Output('precision_recall', 'children')],
        [Input('class_dropdown', 'value')]
    )
    def update_confusion_matrix(selected_class):
        # Get true labels and predictions
        y_true = predictions_df['y_test']
        y_pred = predictions_df['predictions']
        
        # Debug information
        class_info = f"Selected Class: {selected_class}, Instances: {sum(y_true == selected_class)}, Predicted: {sum(y_pred == selected_class)}"
        print(class_info)
        
        # Binarize the labels for the selected class (one vs rest)
        y_true_bin = (y_true == selected_class).astype(int)
        y_pred_bin = (y_pred == selected_class).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true_bin, y_pred_bin)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Create labels for the matrix
        labels = ['Negative (Other Classes)', 'Positive (Selected Class)']
        
        # Create confusion matrix figure
        fig = ff.create_annotated_heatmap(
            z=cm,
            x=labels,
            y=labels,
            annotation_text=cm,
            colorscale='Blues'
        )
        
        # Update layout
        fig.update_layout(
            title=f'Matriz de Confusión para Clase: {selected_class}',
            xaxis=dict(title='Predicted Label'),
            yaxis=dict(title='True Label'),
            **layout_settings
        )
        
        # Create metrics text with additional debug info
        metrics_text = [
            html.Div([
                html.Div(f"Instancias de clase: {sum(y_true_bin)}, Predicciones de esta clase: {sum(y_pred_bin)}",
                        style={'marginBottom': '10px'}),
                html.Span(f"Precisión: {precision:.3f}", style={'marginRight': '20px'}),
                html.Span(f"Exhaustividad: {recall:.3f}", style={'marginRight': '20px'}),
                html.Span(f"Puntuación F1: {f1_score:.3f}")
            ])
        ]
        
        return fig, metrics_text