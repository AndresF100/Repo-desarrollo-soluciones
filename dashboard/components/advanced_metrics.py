import dash
from dash import dcc  # Changed from import dash_core_components as dcc
from dash import html  # Changed from import dash_html_components as html
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
from dash.dependencies import Input, Output
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from utils.data_loader import colors, layout_settings

def get_advanced_metrics_layout(predictions_df):
    """Return the layout for advanced model metrics"""
    return html.Div([
        # Global Metrics Section
        html.Div([
            html.H3('Métricas Globales del Modelo',
                    style={'textAlign': 'center', 'fontFamily': 'Segoe UI', 'color': colors['text'], 'marginBottom': '20px'}),
            
            # Metrics Cards
            html.Div([
                html.Div([
                    html.H4("Precisión Global", style={'color': colors['text'], 'textAlign': 'center'}),
                    html.Div(id="global_accuracy", style={'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center', 'color': colors['text']}),
                ], style={'backgroundColor': '#3498db', 'padding': '20px', 'borderRadius': '10px', 'flex': '1', 'marginRight': '10px'}),
                
                html.Div([
                    html.H4("Precisión Balanceada", style={'color': colors['text'], 'textAlign': 'center'}),
                    html.Div(id="balanced_accuracy", style={'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center', 'color': colors['text']}),
                ], style={'backgroundColor': '#2ecc71', 'padding': '20px', 'borderRadius': '10px', 'flex': '1', 'marginRight': '10px'}),
                
                html.Div([
                    html.H4("F1-Score Macro", style={'color': colors['text'], 'textAlign': 'center'}),
                    html.Div(id="f1_score", style={'fontSize': '24px', 'fontWeight': 'bold', 'textAlign': 'center', 'color': colors['text']}),
                ], style={'backgroundColor': '#e74c3c', 'padding': '20px', 'borderRadius': '10px', 'flex': '1'}),
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'margin': '20px 0'})
            
        ], style={'backgroundColor': colors['panel'], 'padding': '20px', 'borderRadius': '10px', 'width': '90%', 'margin': '20px auto'}),
        
        # Class Distribution Comparison
        html.Div([
            html.H3('Distribución de Clases: Real vs. Predicciones',
                    style={'textAlign': 'center', 'fontFamily': 'Segoe UI', 'color': colors['text'], 'marginBottom': '20px'}),
            dcc.Graph(id='class_distribution'),
        ], style={'backgroundColor': colors['panel'], 'padding': '20px', 'borderRadius': '10px', 'width': '90%', 'margin': '20px auto'}),
        
        # ROC Curve
        html.Div([
            html.H3('Curva ROC por Clase',
                    style={'textAlign': 'center', 'fontFamily': 'Segoe UI', 'color': colors['text'], 'marginBottom': '20px'}),
            dcc.Graph(id='roc_curve'),
        ], style={'backgroundColor': colors['panel'], 'padding': '20px', 'borderRadius': '10px', 'width': '90%', 'margin': '20px auto'})
        
    ])

def register_advanced_metrics_callbacks(app, predictions_df):
    # Global Metrics Callback
    @app.callback(
        [Output('global_accuracy', 'children'),
         Output('balanced_accuracy', 'children'),
         Output('f1_score', 'children')],
        [Input('class_dropdown', 'value')]  # Using class_dropdown as a dummy input to trigger the callback
    )
    def update_global_metrics(dummy):
        # Calculate global metrics
        accuracy = accuracy_score(predictions_df['y_test'], predictions_df['predictions'])
        balanced_acc = balanced_accuracy_score(predictions_df['y_test'], predictions_df['predictions'])
        macro_f1 = f1_score(predictions_df['y_test'], predictions_df['predictions'], average='macro')
        
        return f"{accuracy:.3f}", f"{balanced_acc:.3f}", f"{macro_f1:.3f}"
    
    # Class Distribution Callback
    @app.callback(
        Output('class_distribution', 'figure'),
        [Input('class_dropdown', 'value')]  # Using class_dropdown as a dummy input
    )
    def update_class_distribution(dummy):
        # Count occurrences of each class in actual and predicted data
        actual_counts = predictions_df['y_test'].value_counts().sort_index()
        predicted_counts = predictions_df['predictions'].value_counts().sort_index()
        
        # Create a combined dataframe for plotting
        classes = sorted(set(actual_counts.index) | set(predicted_counts.index))
        distribution_df = pd.DataFrame({
            'Clase': classes,
            'Actual': [actual_counts.get(cls, 0) for cls in classes],
            'Predicción': [predicted_counts.get(cls, 0) for cls in classes]
        })
        
        fig = go.Figure(data=[
            go.Bar(name='Actual', x=distribution_df['Clase'], y=distribution_df['Actual']),
            go.Bar(name='Predicción', x=distribution_df['Clase'], y=distribution_df['Predicción'])
        ])
        
        fig.update_layout(
            barmode='group',
            title='Distribución de Clases: Valores Reales vs. Predicciones',
            xaxis=dict(title='Clase'),
            yaxis=dict(title='Cantidad'),
            **layout_settings,
            legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)')
        )
        return fig
    
    # ROC Curve Callback
    @app.callback(
        Output('roc_curve', 'figure'),
        [Input('class_dropdown', 'value')]
    )
    def update_roc_curve(selected_class):
        # Get true labels and predictions
        y_true = predictions_df['y_test']
        y_pred = predictions_df['predictions']
        
        # Binarize for one-vs-rest
        y_true_bin = (y_true == selected_class).astype(int)
        y_pred_bin = (y_pred == selected_class).astype(int)
        
        # Calculate ROC curve
        try:
            fpr, tpr, _ = roc_curve(y_true_bin, y_pred_bin)
            roc_auc = auc(fpr, tpr)
            
            # Create ROC curve
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=fpr, y=tpr, 
                                 mode='lines',
                                 name=f'ROC curve (area = {roc_auc:.3f})'))
            fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], 
                                 mode='lines', 
                                 line=dict(dash='dash', color='gray'),
                                 name='Random Classifier'))
            
            fig.update_layout(
                title=f'Curva ROC para Clase: {selected_class}',
                xaxis=dict(title='Tasa de Falsos Positivos'),
                yaxis=dict(title='Tasa de Verdaderos Positivos'),
                **layout_settings,
                legend=dict(x=0, y=1, bgcolor='rgba(0,0,0,0)')
            )
            return fig
        except Exception as e:
            # Return empty figure with error message if calculation fails
            fig = go.Figure()
            fig.add_annotation(
                text=f"Error al calcular la curva ROC: {str(e)}",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=0.5,
                font=dict(color=colors['text'], size=16)
            )
            fig.update_layout(**layout_settings)
            return fig