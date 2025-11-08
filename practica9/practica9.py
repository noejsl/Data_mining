import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re

def load_data(filepath):
    """Carga el dataset CSV."""
    df = pd.read_csv(filepath,  encoding='utf-8')
    return df

def combine_text(df, column="Description"):
    """Combina texto de una columna en un solo string y limpia caracteres raros."""
    text = " ".join(df[column].dropna().astype(str))

    # Limpieza
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-ZáéíóúüñÁÉÍÓÚÜÑ\s]', ' ', text) 
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()

    stopwords = {'the', 'and', 'for', 'with', 'this', 'that', 'you', 'your', 
                 'official', 'video', 'music', 'channel', 'subscribe', 'follow'}
    words = [word for word in text.split() if word not in stopwords and len(word) > 2]
    text = " ".join(words)

    return text

def create_wordcloud(text, title="Word Cloud"):
    """Genera y muestra una nube de palabras limpia."""
    wordcloud = WordCloud(
        width=900,
        height=500,
        background_color='white',
        colormap='plasma',
        max_words=150,
        stopwords=set() 
    ).generate(text)

    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=16)
    plt.show()

def main():
    filepath = "../practica2/Songs_Dataset_Clean.csv"

    print("Cargando DataSet")
    df = load_data(filepath)

    print("Combinando texto de la columna descripción")
    text_data = combine_text(df)

    print("Generando World Cloud")
    create_wordcloud(text_data)
    
if __name__ == "__main__":
    main()