import pandas as pd
import base64
from analisis_texto import generate_wordcloud

def load_data():
    """Load and prepare all data needed for the dashboard"""
    # Load the main dataset
    df = pd.read_csv('dashboard/data_for_visuals.csv.xls')
    
    # Prepare day column
    df['day'] = pd.to_datetime(df['fecha_siniestro_igdacmlmasolicitudes']).dt.day
    
    # Load predictions data
    predictions_df = pd.read_csv('dashboard/predictions.csv')
    
    return df, predictions_df

def generate_and_encode_wordcloud(df, output_path="wordcloud.png"):
    """Generate wordcloud and return base64 encoded image"""
    # Generate wordcloud
    wordcloud_result = generate_wordcloud(df, output_path=output_path, show_plot=False)
    
    # Encode the wordcloud image
    encoded_image = base64.b64encode(open(output_path, 'rb').read()).decode('ascii')
    
    return encoded_image

# Define color scheme used across the dashboard
colors = {
    'background': '#2c3e50',  # Grayish blue
    'text': '#FFFFFF',        # White text for better contrast
    'panel': '#34495e'        # Slightly lighter grayish blue for panels
}

# Common layout settings for charts
layout_settings = {
    'paper_bgcolor': colors['panel'],
    'plot_bgcolor': colors['panel'],
    'font': {'color': colors['text'], 'family': 'Segoe UI'},
    'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40}
}