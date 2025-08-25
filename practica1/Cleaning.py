
# Dataset Cleaning Script - Songs_Dataset
# Este script realiza la limpieza básica del dataset de canciones:
# - Llena valores faltantes en Description, Likes y Comments
# - Convierte tipos de datos a los correctos
# - Normaliza fechas y elimina duplicados

import pandas as pd

# 1. Cargando dataset
# Se usa encoding 'latin1' para evitar errores de caracteres
df = pd.read_csv("Songs_Dataset.csv", encoding='latin1')


# 2. Limpiando columna Description
# Llenando valores faltantes con texto predeterminado
df['Description'] = df['Description'].fillna('No description')


# 3. Limpiando y normalizando Likes
# Convertir Likes a float para cálculos
df['Likes'] = df['Likes'].astype(float)

# Calculando tasa promedio de Likes por Views
mask_likes = df['Likes'].notna() & (df['Views'] > 0)
avg_ratio_likes = (df.loc[mask_likes, 'Likes'] / df.loc[mask_likes, 'Views']).mean()

# Rellenando valores faltantes de Likes usando la tasa promedio
df['Likes'] = df['Likes'].fillna((df['Views'] * avg_ratio_likes).round().astype(int))

# 4. Limpiando y normalizando Comments
# Calcular tasa promedio de Comments por Views
mask_comments = df['Comments'].notna() & (df['Views'] > 0)
avg_ratio_comments = (df.loc[mask_comments, 'Comments'] / df.loc[mask_comments, 'Views']).mean()

# Rellenando valores faltantes de Comments usando la tasa promedio
df['Comments'] = df['Comments'].fillna((df['Views'] * avg_ratio_comments).round().astype(int))

# 5. Convertiendo Likes y Comments a enteros
df['Likes'] = df['Likes'].astype(int)
df['Comments'] = df['Comments'].astype(int)

# 6. Convertiendo Release_date a datetime
# Si hay errores de formato, se convierten en NaT
df['Release_date'] = pd.to_datetime(df['Release_date'], errors='coerce')

# 7. Eliminando duplicados
# Se eliminan filas duplicadas de canciones por Artist y Track, conservando la primera
df = df.drop_duplicates(subset=['Artist', 'Track'], keep='first')

# 8. Guardar dataset limpio 
df.to_csv("Songs_Dataset_Clean.csv", index=False, encoding='utf-8')
