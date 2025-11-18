#  IMPORTACIONES GLOBALES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import kruskal

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from imblearn.combine import SMOTETomek
from imblearn.pipeline import Pipeline



# 1. KRUSKAL-WALLI

def perform_kruskal(df, var_dep, var_ind="Album_type"):
    grupos = [
        df[df[var_ind] == g][var_dep].dropna()
        for g in df[var_ind].unique()
    ]
    stat, p = kruskal(*grupos)
    return stat, p


def menu_kruskal(df):
    variables = [
        'Danceability', 'Energy', 'Valence', 'Acousticness',
        'Speechiness', 'Instrumentalness', 'Liveness'
    ]

    while True:
        print("\n--- Menú Kruskal-Wallis ---")
        for i, v in enumerate(variables, 1):
            print(f"{i}. {v}")
        print("8. Salir")

        eleccion = input("Selecciona una opción: ")

        if eleccion == "8":
            print("Saliendo del menú.")
            break

        if not eleccion.isdigit() or not 1 <= int(eleccion) <= 7:
            print("Opción inválida.")
            continue

        var = variables[int(eleccion)-1]
        stat, p = perform_kruskal(df, var)
        print(f"\n{var}: H={stat:.4f}, p={p:.4e}")
        print("Conclusión:",
              "Rechazamos H0 (diferencias significativas)" if p < 0.05
              else "No se rechaza H0")


def print_means_by_album(df):
    variables = [
        'Speechiness','Danceability','Energy','Acousticness',
        'Valence','Instrumentalness','Liveness'
    ]
    print("\n=== Media por tipo de álbum ===")
    print(df.groupby('Album_type')[variables].mean())



# 2. REGRESIÓN LINEAL

def prepare_data(df, features, target):
    df_clean = df.dropna(subset=features + [target]).copy()
    X = np.log1p(df_clean[features])
    y = np.log1p(df_clean[target])
    return X, y


def train_linear_model(X, y):
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    Xtrain_s = scaler.fit_transform(Xtrain)
    Xtest_s = scaler.transform(Xtest)

    model = LinearRegression()
    model.fit(Xtrain_s, ytrain)

    ypred = model.predict(Xtest_s)
    print(f"R² Score: {r2_score(ytest, ypred):.4f}")

    return model, scaler, Xtest, ytest, ypred


def plot_actual_vs_predicted(ytest, ypred, title="Regresión"):
    y_real = np.expm1(ytest)
    y_pred_real = np.expm1(ypred)

    plt.figure(figsize=(7,6))
    plt.scatter(y_real, y_pred_real, alpha=0.5)
    plt.plot([y_real.min(), y_real.max()],
             [y_real.min(), y_real.max()], 'r--')
    plt.xscale("log"); plt.yscale("log")
    plt.title(title)
    plt.xlabel("Real"); plt.ylabel("Predicho")
    plt.show()


def print_coefficients(model, features):
    print("\n=== Coeficientes ===")
    for f, c in zip(features, model.coef_):
        print(f"{f}: {c:.4f}")


# 3. KNN CLASSIFIER

def prepare_knn_data(df, target):
    features = [
        'Danceability', 'Energy', 'Loudness', 'Speechiness',
        'Acousticness', 'Instrumentalness', 'Liveness', 'Valence',
        'Duration_ms', 'Views', 'Likes', 'Comments'
    ]
    df = df.dropna(subset=features + [target])
    return df[features], df[target]


def tune_k(X, y, k_range=range(1,21)):
    f1_scores = []
    for k in k_range:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('balance', SMOTETomek(random_state=42)),
            ('knn', KNeighborsClassifier(n_neighbors=k, weights='distance'))
        ])
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                             random_state=42, stratify=y)
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        f1_scores.append(f1_score(yte, pred, average='macro'))

    best_k = k_range[np.argmax(f1_scores)]
    print(f"Mejor k: {best_k} (F1={max(f1_scores):.4f})")
    return best_k


def train_knn(X, y, k):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('balance', SMOTETomek(random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=k, weights='distance'))
    ])

    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    print("\n=== Reporte KNN ===")
    print(classification_report(yte, pred))
    print("F1-macro:", f1_score(yte, pred, average='macro'))

    return pipe


# 4. K-MEANS CLUSTERING

def scale_features(df, features):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df[features])
    return Xs, scaler


def elbow(Xs, max_k=15):
    inertias = []
    for k in range(2, max_k+1):
        km = KMeans(n_clusters=k, random_state=42).fit(Xs)
        inertias.append(km.inertia_)

    plt.plot(range(2, max_k+1), inertias, marker="o")
    plt.title("Método del Codo")
    plt.xlabel("k"); plt.ylabel("Inercia")
    plt.show()


def silhouette_scan(Xs, max_k=15):
    scores = []
    for k in range(2, max_k+1):
        labels = KMeans(n_clusters=k, random_state=42).fit_predict(Xs)
        scores.append(silhouette_score(Xs, labels))

    plt.plot(range(2, max_k+1), scores, marker="s")
    plt.title("Silhouette Score")
    plt.xlabel("k"); plt.ylabel("Silhouette")
    plt.show()


def train_kmeans(df, Xs, features, k=5):
    km = KMeans(n_clusters=k, random_state=42)
    df["Cluster"] = km.fit_predict(Xs)

    print(f"Silhouette (k={k}): {silhouette_score(Xs, df['Cluster']):.4f}")

    print("\n=== Promedio por cluster ===")
    print(df.groupby("Cluster")[features].mean())

    return km


def plot_pca_clusters(df, Xs):
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(Xs)

    plt.scatter(X2[:,0], X2[:,1], c=df["Cluster"], cmap="viridis", alpha=.6)
    plt.title("Clusters (PCA)")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.show()



# MAIN 


def main():

    print("Cargando dataset")
    df = pd.read_csv(r"../practica2/Songs_Dataset_Clean.csv", encoding='utf-8')

    # KRUSKAL 
    print_means_by_album(df)
    menu_kruskal(df)

    # REGRESIÓN 
    features = [
        'Danceability', 'Energy', 'Speechiness', 'Acousticness',
        'Instrumentalness', 'Liveness', 'Valence',
        'Likes', 'Comments'   # Views es el target, no va
    ]

    X, y = prepare_data(df, features=features, target="Views")
    model, scaler, Xte, yte, ypred = train_linear_model(X, y)

    plot_actual_vs_predicted(yte, ypred, title="Regresión: Predicción de Views")

    print_coefficients(model, features)

    # KNN 
    X, y = prepare_knn_data(df, target="official_video")
    k = tune_k(X, y)
    model_knn = train_knn(X, y, k)

    # KMEANS 
    features = ['Views', 'Likes', 'Comments']
    Xs, sc = scale_features(df, features)

    elbow(Xs)
    silhouette_scan(Xs)

    km = train_kmeans(df, Xs, features, k=5)
    plot_pca_clusters(df, Xs)


if __name__ == "__main__":
    main()
