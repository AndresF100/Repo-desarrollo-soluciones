import pandas as pd
import base64
import os
from analisis_texto import generate_wordcloud

# Get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_data():
    """Load and prepare all data needed for the dashboard"""
    # Use absolute paths with os.path.join
    try:
        # Try relative paths first
        df = pd.read_csv('data_for_visuals.csv.xls')
    except:
        try:
            # Try with BASE_DIR
            df = pd.read_csv(os.path.join(BASE_DIR, 'data_for_visuals.csv.xls'))
        except:
            # Create dummy data if file doesn't exist
            print("Creating dummy data as data file wasn't found")
            df = pd.DataFrame({
                'sector_economico': ['Construcción', 'Manufactura', 'Servicios', 'Minería'],
                'genero': ['Masculino', 'Femenino', 'Masculino', 'Femenino'],
                'tipo_vinculacion': ['Indefinido', 'Fijo', 'Indefinido', 'Fijo'],
                'jornada_trabajo': ['Diurna', 'Nocturna', 'Diurna', 'Nocturna'],
                'descripcion_accidente_igatepmafurat': [
                    'Caída de escalera', 'Golpe con objeto', 'Corte con herramienta', 'Lesión por sobreesfuerzo'
                ]
            })
    
    # Do the same for predictions_df
    try:
        predictions_df = pd.read_csv('predictions.csv')
    except:
        try:
            predictions_df = pd.read_csv(os.path.join(BASE_DIR, 'predictions.csv'))
        except:
            # Create dummy prediction data
            print("Creating dummy prediction data")
            predictions_df = pd.DataFrame({
                'actual': ['Positivo', 'Negativo', 'Positivo', 'Negativo'],
                'predicted': ['Positivo', 'Negativo', 'Negativo', 'Positivo'],
                'probability': [0.8, 0.7, 0.4, 0.3]
            })
    
    # Add day column if date column exists
    if 'fecha_siniestro_igdacmlmasolicitudes' in df.columns:
        df['day'] = pd.to_datetime(df['fecha_siniestro_igdacmlmasolicitudes']).dt.day
    
    return df, predictions_df

def generate_and_encode_wordcloud(df, output_path="dashboard/wordcloud.png"):
    """
    Generate wordcloud and return base64 encoded image.
    Checks if a wordcloud.png file already exists before generating a new one.
    """
    # Check if the wordcloud image already exists
    if os.path.exists(output_path):
        try:
            # Use existing wordcloud file
            print(f"Using existing wordcloud image from {output_path}")
            with open(output_path, 'rb') as image_file:
                encoded_image = base64.b64encode(image_file.read()).decode('ascii')
            return encoded_image
        except Exception as e:
            print(f"Error loading existing wordcloud: {e}")
            # If there's an error loading the file, we'll generate a new one
    
    # If the file doesn't exist or couldn't be loaded, generate a new wordcloud
    print("Generating new wordcloud...")
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Generate wordcloud
    wordcloud_result = generate_wordcloud(df, output_path=output_path, show_plot=False)
    
    # Encode the wordcloud image
    with open(output_path, 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode('ascii')
    
    return encoded_image

# Define color scheme used across the dashboard
colors = {
    'background': '#2c3e50',  # Grayish blue
    'text': '#FFFFFF',        # White text for better contrast
    'panel': '#34495e',        # Slightly lighter grayish blue for panels
    'border': '#FFFFFF',       # White border for better contrast
    'primary': '#3498db',      # Light blue for buttons and links
    'secondary': '#e74c3c',     # Indian red for highlighted items
    'shadow': 'rgba(0, 0, 0, 0.15)'  # Shadow for cards
}

# Common layout settings for charts
layout_settings = {
    'paper_bgcolor': colors['panel'],
    'plot_bgcolor': colors['panel'],
    'font': {'color': colors['text'], 'family': 'Segoe UI'},
    'margin': {'l': 40, 'r': 40, 't': 40, 'b': 40}
}