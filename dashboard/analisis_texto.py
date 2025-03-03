# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 08:31:45 2019

@author: camil
"""
import numpy as np
import pandas as pd
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from collections import Counter

# Download required NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def generate_wordcloud(df, text_column='descripcion_at_igatepmafurat', 
                       output_path="C:/Users/Santiago Calderón/OneDrive - LLA/Documents/Maestria/Proyecto Final/Proyecto de Materia/Taller 1/Repo-desarrollo-soluciones/dash/wordcloud.png",
                       show_plot=True, max_words=200):
    """
    Generate a wordcloud from text data in a DataFrame column
    
    Parameters:
    -----------
    df : pandas DataFrame
        DataFrame containing the text data
    text_column : str
        Name of the column containing text data
    output_path : str
        Path where the wordcloud image will be saved
    show_plot : bool
        Whether to display the plot or not
    max_words : int
        Maximum number of words to include in the wordcloud
        
    Returns:
    --------
    dict : Dictionary with wordcloud, word_counts and processed text
    """
    
    # Ensure we have data in the column
    print(f"Número de filas con texto: {df[text_column].notna().sum()}")

    # Get text from dataframe column
    texto = df[text_column].dropna().tolist()  # Drop NA values

    # Define stopwords - only define once
    stop_words = set(stopwords.words('spanish'))
    # Add custom stopwords - common words that don't add meaning to your analysis
    stop_words.update([
        "realizando", "trabajador", "encontraba", "tener", "cada", "través",
        "cada", "positiva", "tambien", "permitan", "ser", "así", "avaya",
        "tan", "pienso", "sido", "dado", "cómo", "continuidad",
        "negocio", "necesidad", "todas", "niña", "habitual", "labor", "manera", 
        "cliente", "clientes", "haciendo", "ocasionando",
        "realizar", "realizaba", "dia", 
        "durante", "se", "el", "la", "los", "las", "un", "una", 
        "unos", "unas", "al", "del", "lo", "le", "les", "su", "sus", "por",
          "para", "con", "sin"
    ])

    # Process the text
    processed_texts = []
    for i in range(len(texto)):
        if pd.notna(texto[i]):  # Check for NA values
            # Remove punctuation and special characters
            text = re.sub(r'[^\w\s]', '', str(texto[i]))
            # Convert to lowercase
            text = text.lower()
            # Replace accented characters with regular ones
            text = re.sub(r'[áäâà]', 'a', text)
            text = re.sub(r'[éêèë]', 'e', text)
            text = re.sub(r'[íîìï]', 'i', text)
            text = re.sub(r'[óôòö]', 'o', text)
            text = re.sub(r'[úûùü]', 'u', text)
            # Tokenize
            word_tokens = word_tokenize(text)
            # Filter out stopwords
            filtered_sentence = [w for w in word_tokens if w not in stop_words and len(w) > 2]  # Only keep words longer than 2 chars
            processed_texts.append(" ".join(filtered_sentence))

    # Join all processed texts into one string
    all_text = " ".join(processed_texts)

    # Saves the processed text to a file
    with open("processed_text.txt", "w") as file:
        file.write(all_text)

    print(f"Total words after processing: {len(all_text.split())}")
    print(f"Unique words after processing: {len(set(all_text.split()))}")

    # Generate the word cloud with more words
    wordcloud = WordCloud(
        width=2000, 
        height=1000,
        background_color="white",
        max_words=max_words,  # Increase max words
        min_font_size=5,  # Allow smaller fonts
        collocations=True,  # Allow bigrams
        stopwords=stop_words,
        random_state=42  # For reproducibility
    ).generate(all_text)

    # Display the wordcloud if requested
    if show_plot:
        plt.figure(figsize=(16, 8))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.tight_layout(pad=0)
        plt.show()

    # Save the wordcloud
    wordcloud.to_file(output_path)

    # Get word counts
    word_counts = Counter(all_text.split())
    print("Top 20 most common words:")
    for word, count in word_counts.most_common(20):
        print(f"{word}: {count}")
        
    return {
        "wordcloud": wordcloud,
        "word_counts": word_counts,
        "processed_text": all_text
    }


# Execute the function when script is run directly
if __name__ == "__main__":
    # Lee el archivo de excel
    data = pd.read_csv('C:/Users/Santiago Calderón/OneDrive - LLA/Documents/Maestria/Proyecto Final/Proyecto de Materia/Taller 1/Repo-desarrollo-soluciones/dash/data_for_visuals.csv.xls')
    
    # Generate the wordcloud
    result = generate_wordcloud(data)