import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# 1. Prepare time series data 
def prepare_time_series(df, date_col='Release_date', target_col='Views'):
    """
    Predice la popularidad promedio de futuros lanzamientos
    basado en tendencia histórica de qué tan populares son las canciones nuevas
    """
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col, target_col])
    
    # Agrupar por AÑO para tener datos más estables 
    df['year'] = df[date_col].dt.year
    ts = df.groupby('year')[target_col].mean().reset_index()  
    
    ts.columns = ['date', 'target']
    ts['t'] = np.arange(len(ts))  # Time index para regresión
    print("Datos históricos - Vistas promedio por año:")
    for _, row in ts.iterrows():
        print(f"Año {int(row['date'])}: {row['target']:,.0f} vistas promedio")
    
    return ts

# 2. Train linear regression 
def train_linear_regression(ts):
    """Trains linear regression model using time index."""
    X = ts[['t']]
    y = ts['target']
    model = LinearRegression()
    model.fit(X, y)
    
    # Calcular R² para evaluar calidad
    r_squared = model.score(X, y)
    print(f"\nCalidad del modelo: R² = {r_squared:.3f}")
    if r_squared > 0.7:
        print(" Buen ajuste del modelo")
    elif r_squared > 0.4:
        print("Ajuste moderado del modelo")  
    else:
        print("Modelo con bajo poder predictivo")
    
    return model

# 3. Predict future values 
def predict_future(model, ts, periods=3):
    """
    Predice la popularidad promedio para los próximos AÑOS
    """
    last_t = ts['t'].iloc[-1]
    last_year = ts['date'].iloc[-1]
    
    future_t = np.arange(last_t + 1, last_t + periods + 1)
    future_years = [last_year + i for i in range(1, periods + 1)]
    future_pred = model.predict(future_t.reshape(-1, 1))

    future_df = pd.DataFrame({
        'year': future_years, 
        'average_views_pred': future_pred
    })
    return future_df

# 4. Plot results
def plot_forecast(ts, future_df):
    """Displays historical data and forecast."""
    plt.figure(figsize=(10, 5))
    
    # Datos históricos
    plt.plot(ts['date'], ts['target'], label='Vistas promedio históricas', 
             marker='o', linewidth=2, markersize=8)
    
    # Predicciones
    plt.plot(future_df['year'], future_df['average_views_pred'], 
             label='Pronóstico', linestyle='--', marker='x', linewidth=2)
    
    plt.xlabel('Año')
    plt.ylabel('Vistas Promedio')
    plt.title('Pronóstico: Vistas Promedio de Canciones por Año\n(Tendencia de Popularidad)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# Main execution
def main():

    # 1. Load dataset
    df = pd.read_csv(r"../practica2/Songs_Dataset_Clean.csv", encoding='utf-8')


    # 2. Prepare time series 
    ts = prepare_time_series(df, date_col='Release_date', target_col='Views')
    
    if len(ts) < 3:
        print("No hay suficientes años de datos para forecasting")
        return
    
    # 3. Train linear regression
    model = train_linear_regression(ts)
    
    # 4. Predict next 3 años
    future_df = predict_future(model, ts, periods=3)

    # 5. Plot results
    plot_forecast(ts, future_df)

    # 6. Print predictions
    print("\nPRONÓSTICO - Vistas Promedio Esperadas:")
    for _, row in future_df.iterrows():
        print(f"Año {int(row['year'])}: {row['average_views_pred']:,.0f} vistas promedio")
    
    # Interpretación
    trend = "creciente" if model.coef_[0] > 0 else "decreciente"
    print(f"\nTendencia: {trend} ({model.coef_[0]:,.0f} vistas/año)")

if __name__ == "__main__":
    main()