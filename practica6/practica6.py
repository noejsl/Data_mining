import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline

def train_knn(df, target='Album_type', k=5):
    """
    Entrena un modelo KNN con balanceo SMOTETomek y escalado.
    Evalúa usando métricas adecuadas (F1-score, precision, recall).
    """
    features = [
        'Danceability', 'Energy', 'Loudness', 'Speechiness',
        'Acousticness', 'Instrumentalness', 'Liveness',
        'Valence', 'Duration_ms'
    ]
    
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Pipeline escalado con weights='distance'
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('balance', SMOTETomek(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=k, weights='distance'))
    ])
    
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    print("\nEvaluación del modelo KNN")
    print(f"F1-Score ponderado: {f1_score(y_test, y_pred, average='weighted'):.4f}\n")
    print(classification_report(y_test, y_pred, digits=4))
    
    plot_confusion(y_test, y_pred)
    return pipeline

def plot_confusion(y_test, y_pred):
    """Grafica la matriz de confusión."""
    cm = confusion_matrix(y_test, y_pred)
    labels = sorted(y_test.unique())
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión - KNN")
    plt.show()

def tune_k(df, target='Album_type', k_range=range(1, 21)):
    """
    Prueba distintos valores de k y muestra cuál da mejor F1-score ponderado.
    """
    features = [
        'Danceability', 'Energy', 'Loudness', 'Speechiness',
        'Acousticness', 'Instrumentalness', 'Liveness',
        'Valence', 'Duration_ms'
    ]
    
    df = df.dropna(subset=features + [target])
    X = df[features]
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    f1_scores = []
    
    for k in k_range:
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('balance', SMOTETomek(random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=k, weights='distance'))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        f1_scores.append(f1)
    
    plt.figure(figsize=(8,5))
    plt.plot(k_range, f1_scores, marker='o', linewidth=2)
    plt.title("F1-Score ponderado para distintos valores de k")
    plt.xlabel("Número de vecinos (k)")
    plt.ylabel("F1-Score ponderado")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()
    
    best_k = k_range[np.argmax(f1_scores)]
    print(f"Mejor k encontrado: {best_k} (F1 = {max(f1_scores):.4f})")
    return best_k

def main():
    df = pd.read_csv(r"../practica2/Songs_Dataset_Clean.csv", encoding='utf-8')
    
    print("Buscando mejor valor de k...\n")
    best_k = tune_k(df)
    
    print("\nEntrenando modelo final...\n")
    model = train_knn(df, k=best_k)
    
    
if __name__ == "__main__":
    main()
