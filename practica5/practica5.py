import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler


def train_linear_model(df, target='Views'):
    """
    Entrena un modelo lineal para predecir views en base a metricas musicales.
    Devuelve modelo, datos de prueba y predicciones.
    """
    features = ['Danceability','Energy','Loudness','Speechiness',
                'Acousticness','Instrumentalness','Liveness',
                'Valence','Duration_ms']

    df = df.dropna(subset=features + [target])  # eliminar filas con nulos
    X = df[features]
    y = np.log1p(df[target])  # log-transformar target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalado de características
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Entrenamiento del modelo
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Evaluación del modelo
    y_pred = model.predict(X_test_scaled)
    print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

    return model, X_test, y_test, y_pred, features


def plot_results(y_test, y_pred):
    """Comparamos de valores reales vs predichos en escala original."""
    y_test_orig = np.expm1(y_test)
    y_pred_orig = np.expm1(y_pred)

    plt.figure(figsize=(8,6))
    plt.scatter(y_test_orig, y_pred_orig, alpha=0.5)
    plt.plot([y_test_orig.min(), y_test_orig.max()],
             [y_test_orig.min(), y_test_orig.max()], 'r--', lw=2)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel("Actual Views (log scale)")
    plt.ylabel("Predicted Views (log scale)")
    plt.title("Actual vs Predicted Views")
    plt.show()
    
    

def plot_coefficients(model, features):
    """Mostramos los coeficientes del modelo en tabla y gráfico de barras."""
    coef_df = pd.DataFrame({"Feature": features, "Coefficient": model.coef_}).sort_values(by="Coefficient", ascending=False)
    print(coef_df)
    plt.figure(figsize=(10,6))
    sns.barplot(x="Coefficient", y="Feature", data=coef_df)
    plt.title("Coeficientes del modelo lineal")
    plt.show()
    

def main():
    df = pd.read_csv(r"../practica2/Songs_Dataset_Clean.csv", encoding='utf-8')
    model, X_test, y_test, y_pred, features = train_linear_model(df, target='Views')
    plot_results(y_test, y_pred)
    plot_coefficients(model, features)
    
if __name__ == "__main__":
    main()