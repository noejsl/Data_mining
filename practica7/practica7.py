import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

def scale_features(df, features):
    """Escala las características numéricas usando StandardScaler."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    return X_scaled, scaler

def plot_elbow(X_scaled, max_k=10):
    """Usa el método del codo para sugerir un número de clusters."""
    inertia = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        inertia.append(kmeans.inertia_)
    
    plt.figure(figsize=(8,5))
    plt.plot(K_range, inertia, marker='o', linestyle='-', color='royalblue')
    plt.title("Método del Codo (Elbow Method)")
    plt.xlabel("Número de clusters (k)")
    plt.ylabel("Inercia")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
def plot_clusters(df):
    """Grafica Energy vs Danceability con colores por cluster."""
    plt.figure(figsize=(7,6))
    sns.scatterplot(
        data=df, x='Energy', y='Danceability',
        hue='Cluster', palette='Set2', s=80, alpha=0.8
    )
    plt.title("Visualización de Clusters (Energy vs Danceability)")
    plt.xlabel("Energy")
    plt.ylabel("Danceability")
    plt.legend(title="Cluster")
    plt.show()
    

def train_kmeans(X_scaled, df, features, n_clusters=4):
    """Entrena K-Means, evalúa con silhouette y añade etiquetas al DataFrame."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    
    score = silhouette_score(X_scaled, df['Cluster'])
    print(f"Silhouette Score para k={n_clusters}: {score:.4f}")
    
    # Resumen de los clusters
    summary = df.groupby('Cluster')[features].mean().round(3)
    print("\nResumen de características promedio por cluster:\n")
    print(summary)
    
    plot_clusters(df)
    return kmeans

def main():
    df = pd.read_csv(r"../practica2/Songs_Dataset_Clean.csv", encoding='utf-8')
    features = [
        'Danceability', 'Energy', 'Loudness', 'Speechiness',
        'Acousticness', 'Instrumentalness', 'Liveness', 'Valence'
    ]
    X_scaled, scaler = scale_features(df, features)
    
    print("\n Buscando un numero óptimo de clusters (método del codo)")
    plot_elbow(X_scaled, max_k=10)
    n_clusters = 4  # Basado en la observación del gráfico del codo
    model = train_kmeans(X_scaled, df, features, n_clusters=4)
    
    
if __name__ == "__main__":
    main()
