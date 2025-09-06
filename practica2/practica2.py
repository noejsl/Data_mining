import pandas as pd        
import numpy as np         
import graphviz            
from IPython.display import display

def load_data():
    """Cargamos el dataset.
    """
    filepath = r"Songs_Dataset_Clean.csv"
    df = pd.read_csv(filepath, encoding='utf-8')
    print(df.head())
    
    return df

def explore_data_structure(df):
    """Exploramos la estructura básica del dataset mostrando dimensiones, 
    tipos de datos y información general.
    """
    print(f"Dimensiones del dataset: {df.shape}")
    print(f"\nNombres de columnas: {list(df.columns)}")
    print(f"\nTipos de datos:\n{df.dtypes}")
    print(f"\nInformación general:")
    df.info()

def basic_statistics(df):
    """Estadísticas descriptivas básicas para variables numéricas: Media, mediana, desviación estándar, min, max, cuartiles.
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    return df[numeric_columns].describe().T


def advanced_statistics(df):
    """Estadísticas descriptivas avanzadas:Varianza, rango, coeficiente de variación, asimetría y curtosis.
    """
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    advanced_stats = pd.DataFrame({
        'Varianza': df[numeric_columns].var(),
        'Rango': df[numeric_columns].max() - df[numeric_columns].min(),
        'Coef. Variación (%)': (df[numeric_columns].std() / df[numeric_columns].mean()) * 100,
        'Asimetría': df[numeric_columns].skew(),
        'Curtosis': df[numeric_columns].kurtosis()
    })
    
    return advanced_stats

def plot_entities_relations():
    print("Diagrama de Entidades y Relaciones")
    dot = graphviz.Digraph()
    dot.attr(rankdir='TB')  
    
    #Nodos del diagrama
    dot.node('Artist', '''<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
            <TR><TD BGCOLOR="#e6f3ff"><B>ARTISTA</B></TD></TR>
            <TR><TD ALIGN="LEFT">Artist (PK)</TD></TR>
            <TR><TD ALIGN="LEFT">Url_spotify</TD></TR>
        </TABLE>>''')
    
    dot.node('Album', '''<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
            <TR><TD BGCOLOR="#e6f3ff"><B>ÁLBUM</B></TD></TR>
            <TR><TD ALIGN="LEFT">Album (PK)</TD></TR>
            <TR><TD ALIGN="LEFT">Album_type</TD></TR>
        </TABLE>>''')
    
    dot.node('Track', '''<
        <TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0">
            <TR><TD BGCOLOR="#e6f3ff"><B>CANCIÓN</B></TD></TR>
            <TR><TD ALIGN="LEFT">Track</TD></TR>
            <TR><TD ALIGN="LEFT">Uri (PK)</TD></TR>
            <TR><TD ALIGN="LEFT">Danceability</TD></TR>
            <TR><TD ALIGN="LEFT">Energy</TD></TR>
            <TR><TD ALIGN="LEFT">Loudness</TD></TR>
            <TR><TD ALIGN="LEFT">Valence</TD></TR>
        </TABLE>
    >''')

    # Relaciones con cardinalidades
    dot.edge('Artist', 'Album', label='crea\n(1,N)')
    dot.edge('Album', 'Track', label='contiene\n(1,N)')

    try:
        display(dot)
    except:
        print("No se puede mostrar el diagrama aquí. ")
    

def group_by_entities(df):
    """Agrupamos datos por entidad y calculamos estadísticas útiles:
    - por artista: promedio y dispersión de métricas musicales
    - por álbum: número de canciones y promedio de métricas
    """
    # Estadísticas por artista: promedio, desviación estándar, mínimo y máximo
    artist_stats = df.groupby('Artist', sort=False)[['Danceability','Energy','Loudness','Valence']].agg(['mean','std','min','max'])
    
    # Estadísticas por álbum: promedio y número de canciones
    album_stats = df.groupby('Album', sort=False)[['Duration_ms','Danceability','Energy','Valence']].agg(['mean','count'])

    return artist_stats, album_stats


def top_songs_by_views(df, n=10):
    """Retorna las canciones con más visitas en YouTube.
     """ 
    top_views = (
        df[['Track', 'Artist', 'Views']]
        .sort_values(by='Views', ascending=False)
        .head(n)
        .reset_index(drop=True)
    )
    
    return top_views


def main():
    df = load_data()

    print("\nExploración de datos")
    explore_data_structure(df)

    print("\nEstadísticas básicas")
    print(basic_statistics(df))

    print("\nEstadísticas avanzadas")
    print(advanced_statistics(df))

    print("\nDiagrama de entidades y relaciones")
    plot_entities_relations()

    print("\nEstadísticas por entidad")
    artist_stats, album_stats = group_by_entities(df)
    print("\nArtistas:\n", artist_stats.head())
    print("\nÁlbumes:\n", album_stats.head())

    print("\nTop canciones por views")
    print(top_songs_by_views(df, n=10))

main()