from scipy.stats import kruskal
import pandas as pd
import numpy as np

# Cargar dataset
filepath = r"../practica2/Songs_Dataset_Clean.csv"
df = pd.read_csv(filepath, encoding='utf-8')

# Función: Kruskal-Wallis Test
def perform_kruskal(df, variable_dependiente, variable_independiente="Album_type"):
    """
    Realiza la prueba no paramétrica de Kruskal-Wallis para comparar
    una variable cuantitativa (dependiente) entre varios grupos definidos
    por una variable categórica (independiente).
    
    Parámetros:
    df : DataFrame
    variable_dependiente:Nombre de la columna cuantitativa a comparar (ej. 'Danceability').
    variable_independiente : Nombre de la columna categórica que define los grupos (default = 'Album_type').
    
    Retorna:
    stat : Estadístico H de Kruskal-Wallis.
    p p-valor asociado a la prueba.
    
    Nota:
    - Hipótesis nula (H0): Las distribuciones de la variable dependiente 
      son iguales en todos los grupos.
    - Hipótesis alternativa (H1): Al menos un grupo difiere de los demás.
    """
    
    # Crear una lista con los valores de la variable dependiente separados por cada categoría de la variable independiente.
    grupos = [df[df[variable_independiente] == g][variable_dependiente].dropna()
              for g in df[variable_independiente].unique()]
    
    # Prueba de Kruskal-Wallis
    stat, p = kruskal(*grupos)
    return stat, p


# Menú seleccionar la variable dependiente a analizar
while True:
    print("\nMenú Kruskal-Wallis Test:")
    print("1. Comparación de Danceability entre tipos de álbum")
    print("2. Comparación de Energy entre tipos de álbum")
    print("3. Comparación de Valence entre tipos de álbum")
    print("4. Comparación de Acousticness entre tipos de álbum")
    print("5. Salir")
    
    choice = input("Selecciona una opción: ")
    
    if choice == '1':
        stat, p = perform_kruskal(df, 'Danceability')
        print(f"\nDanceability - Estadístico H: {stat:.4f}, p-valor: {p:.4e}")
        print("Conclusión:", "Rechazamos H0 (existen diferencias)" if p < 0.05 else "No se rechaza H0 (no hay diferencias)")
        
    elif choice == '2':
        stat, p = perform_kruskal(df, 'Energy')
        print(f"\nEnergy - Estadístico H: {stat:.4f}, p-valor: {p:.4e}")
        print("Conclusión:", "Rechazamos H0 (existen diferencias)" if p < 0.05 else "No se rechaza H0 (no hay diferencias)")
        
    elif choice == '3':
        stat, p = perform_kruskal(df, 'Valence')
        print(f"\nValence - Estadístico H: {stat:.4f}, p-valor: {p:.4e}")
        print("Conclusión:", "Rechazamos H0 (existen diferencias)" if p < 0.05 else "No se rechaza H0 (no hay diferencias)")
        
    elif choice == '4':
        stat, p = perform_kruskal(df, 'Acousticness')
        print(f"\nAcousticness - Estadístico H: {stat:.4f}, p-valor: {p:.4e}")
        print("Conclusión:", "Rechazamos H0 (existen diferencias)" if p < 0.05 else "No se rechaza H0 (no hay diferencias)")
        
    elif choice == '5':
        print("Saliendo del menú.")
        break
        
    else:
        print("Opción no válida. Por favor, elige una opción válida.")
