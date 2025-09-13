import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Funciones de visualización

def plot_album_type_pie(df):
    """Muestra un gráfico de pastel con la proporción de álbumes y singles."""
    counts = df['Album_type'].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=90)
    plt.title('Proporción de Album vs Single')
    plt.show()


def plot_views_likes_scatter(df):
    """Muestra un scatter plot con la relación entre vistas (Views) y likes (Likes)."""
    plt.scatter(df['Views'], df['Likes'], alpha=0.5)
    plt.title('Relación entre Views y Likes (YouTube)')
    plt.xlabel('Views')
    plt.ylabel('Likes')
    plt.show()


def plot_channel_views_bar(df, top_n=10):
    """Muestra un gráfico de barras con los canales que tienen más vistas acumuladas."""
    views_by_channel = df.groupby('Channel')['Views'].sum().sort_values(ascending=False).head(top_n)
    views_by_channel.plot(kind='bar')
    plt.title(f'Top {top_n} Canales con más Vistas')
    plt.xlabel('Channel')
    plt.ylabel('Total de Views')
    plt.xticks(rotation=45, ha='right')
    plt.show()


def plot_audio_features_hist(df, features=None):
    """Muestra histogramas de varias métricas de audio para comparar sus distribuciones."""
    if features is None:
        features = ['Danceability', 'Energy', 'Valence', 'Acousticness', 'Instrumentalness']
    
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(features):
        plt.subplot(2, 3, i+1)
        plt.hist(df[feature], bins=20, edgecolor='black')
        plt.title(f'Distribución de {feature}')
        plt.xlabel(feature)
        plt.ylabel('Frecuencia')
    plt.tight_layout()
    plt.show()


def plot_audio_pairplot(df):
    """Muestra un pairplot con relaciones entre métricas de audio clave."""
    features = ['Danceability', 'Energy', 'Valence', 'Loudness']
    sns.pairplot(df[features], diag_kind='hist')
    plt.suptitle("Relaciones entre métricas de audio", y=1.02)
    plt.show()


def plot_danceability_by_albumtype(df):
    """Muestra un boxplot de Danceability separado por tipo de álbum (album vs single)."""
    sns.boxplot(x='Album_type', y='Danceability', data=df)
    plt.title('Danceability por tipo de Álbum')
    plt.show()


def plot_views_hist_log(df):
    """Muestra un histograma de Views en escala logarítmica."""
    plt.hist(df['Views'].dropna().apply(lambda x: np.log1p(x)), bins=30, edgecolor='black')
    plt.title('Distribución de Views (escala log)')
    plt.xlabel('log(Views)')
    plt.ylabel('Frecuencia')
    plt.show()


def plot_comments_vs_views(df):
    """Muestra un scatter plot con la relación entre Comments y Views en escala log-log."""
    plt.scatter(df['Views'], df['Comments'], alpha=0.5)
    plt.title('Relación entre Views y Comments (YouTube)')
    plt.xlabel('Views')
    plt.ylabel('Comments')
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


def plot_top10_songs_likes(df):
    """Muestra un gráfico de barras con las 10 canciones con más Likes."""
    top10 = df.nlargest(10, 'Likes')[['Track', 'Likes']].set_index('Track')
    top10.plot(kind='bar', legend=False)
    plt.title('Top 10 Canciones con más Likes')
    plt.ylabel('Likes')
    plt.xticks(rotation=45, ha='right')
    plt.show()


def plot_top10_artists_views(df):
    """Muestra un gráfico de barras con los 10 artistas con más Views acumulados."""
    top10_views = df.groupby('Artist')['Views'].sum().sort_values(ascending=False).head(10)
    top10_views.plot(kind='bar')
    plt.title('Top 10 Artistas con más Views acumulados en YouTube')
    plt.ylabel('Total de Views')
    plt.xticks(rotation=45, ha='right')
    plt.show()


# Función principal
def main():
    filepath = r"../practica2/Songs_Dataset_Clean.csv"
    df = pd.read_csv(filepath, encoding='utf-8')
    
    # Llamar a todas las funciones de visualización
    plot_album_type_pie(df)
    plot_views_likes_scatter(df)
    plot_channel_views_bar(df)
    plot_audio_features_hist(df)
    plot_audio_pairplot(df)
    plot_danceability_by_albumtype(df)
    plot_views_hist_log(df)
    plot_comments_vs_views(df)
    plot_top10_songs_likes(df)
    plot_top10_artists_views(df)

main()
