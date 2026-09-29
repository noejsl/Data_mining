# 🎵 Minería de Datos: Spotify + YouTube

Proyecto de la materia de **Minería de Datos** que recorre el proceso completo de análisis sobre un dataset de canciones que combina métricas de audio de **Spotify** con estadísticas de **YouTube**: limpieza, exploración, visualización, pruebas estadísticas, regresión, clasificación, clustering, series de tiempo y texto.

El repositorio se organiza en **9 prácticas** (cada una es un paso del proceso) y un **PIA** (Proyecto Integrador de Aprendizaje) que reúne los modelos principales en un solo script.

---

## 📑 Contenido

- [Dataset](#-dataset)
- [Estructura del repositorio](#-estructura-del-repositorio)
- [Instalación](#-instalación)
- [Cómo ejecutar los scripts](#-cómo-ejecutar-los-scripts)
- [Prácticas](#-prácticas)
- [PIA](#-pia-proyecto-integrador)
- [Resultados destacados](#-resultados-destacados)
- [Notas y limitaciones conocidas](#-notas-y-limitaciones-conocidas)
- [Solución de problemas](#-solución-de-problemas)

---

## 📊 Dataset

| Archivo | Ubicación | Filas × Columnas | Descripción |
|---|---|---|---|
| `Songs_Dataset.csv` | `practica1/` | 9,307 × 25 | Datos originales (codificación `latin1`) |
| `Songs_Dataset_Clean.csv` | `practica1/` | — | Salida de `Cleaning.py` |
| `Songs_Dataset_Clean.csv` | `practica2/` | 8,638 × 25 | **Dataset que usan las prácticas 2 a 9 y el PIA** |

Cada fila es una canción, con **991 artistas** distintos y tres tipos de álbum: `album` (7,123), `single` (1,051) y `compilation` (464).

**Columnas principales**

| Grupo | Columnas |
|---|---|
| Identificación | `Artist`, `Track`, `Album`, `Album_type`, `Uri`, `Url_spotify`, `Url_youtube` |
| Audio (Spotify) | `Danceability`, `Energy`, `Loudness`, `Speechiness`, `Acousticness`, `Instrumentalness`, `Liveness`, `Valence`, `Duration_ms` |
| YouTube | `Title`, `Channel`, `Views`, `Likes`, `Comments`, `Description`, `Licensed`, `official_video` |
| Fecha | `Release_date` |

---

## 🗂 Estructura del repositorio

```
Data_mining/
├── practica1/
│   ├── Cleaning.py                  # Limpieza del dataset
│   ├── Songs_Dataset.csv            # Datos crudos
│   └── Songs_Dataset_Clean.csv
├── practica2/
│   ├── practica2.py                 # Exploración y estadística descriptiva
│   ├── Songs_Dataset_Clean.csv      # ← dataset base del resto del proyecto
│   └── assets/                      # Diagrama Entidad-Relación
├── practica3/
│   ├── practica3.py                 # 10 visualizaciones
│   └── assets/                      # Gráficas generadas
├── practica4/practica4.py           # Prueba de Kruskal-Wallis (menú interactivo)
├── practica5/
│   ├── practica5.py                 # Regresión lineal
│   └── assets/
├── practica6/practica6.py           # Clasificación KNN + SMOTETomek
├── practica7/practica7.py           # Clustering K-Means
├── practica8/practica8.py           # Pronóstico de tendencia
├── practica9/practica9.py           # Nube de palabras
└── PIA/PIA.py                       # Proyecto integrador
```

---

## ⚙️ Instalación

**Requisitos:** Python 3.9 o superior.

```bash
# 1. Clonar el repositorio
git clone https://github.com/noejsl/Data_mining.git
cd Data_mining

# 2. (Recomendado) crear un entorno virtual
python -m venv venv
source venv/bin/activate          # Linux / macOS
venv\Scripts\activate             # Windows

# 3. Instalar dependencias
pip install pandas numpy matplotlib seaborn scipy scikit-learn \
            imbalanced-learn wordcloud graphviz ipython
```

> El repo no incluye un `requirements.txt`; la lista anterior cubre todos los scripts. Si quieres uno, ejecuta `pip freeze > requirements.txt` después de instalar.

**Solo para la práctica 2 (diagrama ER):** además del paquete de Python `graphviz`, necesitas el programa Graphviz instalado en el sistema:

| Sistema | Comando |
|---|---|
| Ubuntu / Debian | `sudo apt install graphviz` |
| macOS | `brew install graphviz` |
| Windows | Descargar desde [graphviz.org/download](https://graphviz.org/download/) y agregar `bin` al `PATH` |

---

## ▶️ Cómo ejecutar los scripts

> ⚠️ **Importante:** los scripts leen el dataset con rutas **relativas** (`../practica2/Songs_Dataset_Clean.csv`). Por eso **debes ejecutar cada script desde su propia carpeta**.

```bash
cd practica5
python practica5.py
```

Todas las gráficas se abren en ventanas de `matplotlib`: cierra cada una para que el script continúe. Si ejecutas en un servidor sin pantalla, usa:

```bash
MPLBACKEND=Agg python practica5.py     # no muestra ventanas
```

Resumen rápido:

| Script | Comando | ¿Interactivo? | Tiempo aprox. |
|---|---|---|---|
| Limpieza | `cd practica1 && python Cleaning.py` | No | segundos |
| Exploración | `cd practica2 && python practica2.py` | No | segundos |
| Visualizaciones | `cd practica3 && python practica3.py` | Cierra 10 ventanas | segundos |
| Kruskal-Wallis | `cd practica4 && python practica4.py` | **Sí (menú)** | instantáneo |
| Regresión | `cd practica5 && python practica5.py` | Cierra 2 ventanas | segundos |
| KNN | `cd practica6 && python practica6.py` | Cierra 2 ventanas | ~1 min |
| K-Means | `cd practica7 && python practica7.py` | Cierra 2 ventanas | segundos |
| Pronóstico | `cd practica8 && python practica8.py` | Cierra 1 ventana | segundos |
| Nube de palabras | `cd practica9 && python practica9.py` | Cierra 1 ventana | segundos |
| PIA | `cd PIA && python PIA.py` | **Sí (menú)** + varias ventanas | ~1–2 min |

---

## 📚 Prácticas

### Práctica 1 · Limpieza de datos

**Script:** `practica1/Cleaning.py`

Lee `Songs_Dataset.csv` (con `encoding='latin1'`) y aplica:

1. Rellena `Description` vacía con `"No description"`.
2. Imputa `Likes` y `Comments` faltantes usando la **tasa promedio por vista** (`Likes/Views`, `Comments/Views`) multiplicada por las `Views` de cada canción.
3. Convierte `Likes` y `Comments` a enteros.
4. Convierte `Release_date` a `datetime` (los errores se vuelven `NaT`).
5. Elimina duplicados por (`Artist`, `Track`), conservando el primero.
6. Guarda `Songs_Dataset_Clean.csv` en UTF-8.

```bash
cd practica1
python Cleaning.py
```

**Salida:** `practica1/Songs_Dataset_Clean.csv` (se sobrescribe si ya existe).

---

### Práctica 2 · Exploración y estadística descriptiva

**Script:** `practica2/practica2.py`

Funciones incluidas: `load_data`, `explore_data_structure`, `basic_statistics`, `advanced_statistics`, `plot_entities_relations`, `group_by_entities`, `top_songs_by_views`.

Imprime:
- Dimensiones, tipos de datos e información general.
- Estadísticas básicas (`describe`) y avanzadas (varianza, rango, coeficiente de variación, asimetría, curtosis).
- **Diagrama Entidad-Relación** con Graphviz: `Artista (1,N) → Álbum (1,N) → Canción`.
- Estadísticas agrupadas por artista y por álbum.
- **Top 10 canciones por vistas** de YouTube.

```bash
cd practica2
python practica2.py
```

![Diagrama Entidad-Relación](practica2/assets/Diagrama%20Entidad-Relaciones.png)

> El diagrama se muestra con `IPython.display`, así que se ve directamente en **Jupyter/VS Code Interactive**. En una terminal normal se imprimirá el aviso *"No se puede mostrar el diagrama aquí"*; el resto del script funciona igual.

---

### Práctica 3 · Visualización de datos

**Script:** `practica3/practica3.py`

Genera 10 gráficas, una por función:

| Función | Gráfica |
|---|---|
| `plot_album_type_pie` | Proporción álbum vs. single |
| `plot_views_likes_scatter` | Dispersión Views vs. Likes |
| `plot_channel_views_bar` | Top 10 canales por vistas acumuladas |
| `plot_audio_features_hist` | Histogramas de métricas de audio |
| `plot_audio_pairplot` | Pairplot de Danceability, Energy, Valence, Loudness |
| `plot_danceability_by_albumtype` | Boxplot de Danceability por tipo de álbum |
| `plot_views_hist_log` | Histograma de Views (escala log) |
| `plot_comments_vs_views` | Dispersión Comments vs. Views (log-log) |
| `plot_top10_songs_likes` | Top 10 canciones con más Likes |
| `plot_top10_artists_views` | Top 10 artistas con más Views |

```bash
cd practica3
python practica3.py
```

Las imágenes de referencia están en `practica3/assets/`. Por ejemplo:

![Proporción de Album vs Single](practica3/assets/Proporci%C3%B3n%20de%20Album%20vs%20Single.png)

---

### Práctica 4 · Prueba de Kruskal-Wallis

**Script:** `practica4/practica4.py`

Prueba no paramétrica para saber si una métrica de audio **difiere según el tipo de álbum** (`album`, `single`, `compilation`).
- **H₀:** las distribuciones son iguales en todos los grupos.
- **H₁:** al menos un grupo difiere.
- Se rechaza H₀ cuando `p < 0.05`.

```bash
cd practica4
python practica4.py
```

Aparece un menú; escribe el número y presiona Enter:

```
1. Comparación de Danceability entre tipos de álbum
2. Comparación de Energy entre tipos de álbum
3. Comparación de Valence entre tipos de álbum
4. Comparación de Acousticness entre tipos de álbum
5. Salir
```

---

### Práctica 5 · Regresión lineal

**Script:** `practica5/practica5.py`

Predice las **Views** (con transformación `log1p`) a partir de 9 métricas de audio. Usa división 80/20, `StandardScaler` y `LinearRegression`. Muestra el R², una gráfica *Actual vs. Predicted* y los coeficientes del modelo.

```bash
cd practica5
python practica5.py
```

| | |
|---|---|
| ![Actual vs Predicted](practica5/assets/Actual%20vs%20Predicted%20Views.png) | ![Coeficientes](practica5/assets/Coeficientes%20del%20modelo%20lineal.png) |

---

### Práctica 6 · Clasificación con KNN

**Script:** `practica6/practica6.py`

Clasifica el **tipo de álbum** (`album` / `single` / `compilation`) a partir de las métricas de audio. Como las clases están muy desbalanceadas, usa un pipeline:
`StandardScaler → SMOTETomek → KNN (weights='distance')`.

1. `tune_k` prueba k de 1 a 20 y elige el de mejor F1 ponderado.
2. `train_knn` entrena el modelo final e imprime el reporte de clasificación y la matriz de confusión.

```bash
cd practica6
python practica6.py
```

---

### Práctica 7 · Clustering con K-Means

**Script:** `practica7/practica7.py`

Agrupa las canciones según 8 métricas de audio (estandarizadas). Muestra el **método del codo** para k = 2…10, entrena con **k = 4**, calcula el *silhouette score*, imprime el promedio de cada característica por clúster y grafica Energy vs. Danceability coloreado por clúster.

```bash
cd practica7
python practica7.py
```

> Para cambiar el número de clústers edita `n_clusters` en `main()` (nota: la llamada a `train_kmeans` usa `n_clusters=4` fijo).

---

### Práctica 8 · Pronóstico de tendencia

**Script:** `practica8/practica8.py`

Calcula las **vistas promedio por año de lanzamiento** y ajusta una regresión lineal sobre el índice de tiempo para proyectar los **próximos 3 años**. Reporta el R², la tendencia (creciente/decreciente) y grafica histórico + pronóstico.

```bash
cd practica8
python practica8.py
```

Para cambiar el horizonte, modifica `periods=3` en la llamada a `predict_future`.

---

### Práctica 9 · Nube de palabras

**Script:** `practica9/practica9.py`

Analiza la columna `Description`: une todos los textos, elimina URLs y caracteres especiales, pasa a minúsculas, quita *stopwords* (`the`, `and`, `official`, `video`, `subscribe`…) y palabras de menos de 3 letras, y genera una **WordCloud** con las 150 palabras más frecuentes.

```bash
cd practica9
python practica9.py
```

---

## 🏁 PIA (Proyecto Integrador)

**Script:** `PIA/PIA.py`

Integra en un solo flujo las técnicas de las prácticas anteriores, con algunas mejoras:

| Etapa | Qué hace |
|---|---|
| **1. Kruskal-Wallis** | Imprime medias por tipo de álbum y abre un menú con **7 variables** (Danceability, Energy, Valence, Acousticness, Speechiness, Instrumentalness, Liveness). Opción `8` para salir. |
| **2. Regresión lineal** | Predice `Views` con métricas de audio + `Likes` + `Comments`, aplicando `log1p` a características y objetivo. Grafica real vs. predicho e imprime coeficientes. |
| **3. KNN** | Clasifica si un video es **oficial** (`official_video`) usando audio + Views/Likes/Comments. Busca el mejor k (1–20) con **F1 macro** y muestra el reporte. |
| **4. K-Means** | Agrupa por `Views`, `Likes` y `Comments`. Muestra codo y silhouette para k = 2…15, entrena con **k = 5** y visualiza los clústers con **PCA** en 2D. |

```bash
cd PIA
python PIA.py
```

1. Verás las medias por tipo de álbum y el menú de Kruskal-Wallis. Elige variables (1–7) o escribe `8` para continuar.
2. Se abren las gráficas una a una (regresión, codo, silhouette, PCA); ciérralas para avanzar.
3. Los reportes de KNN y clústers se imprimen en la terminal.

---

## 📈 Resultados destacados

Valores obtenidos al ejecutar los scripts con el dataset incluido (`random_state=42`).

| Práctica | Resultado |
|---|---|
| **4 · Kruskal-Wallis** | Se rechaza H₀ en las 4 variables (p < 0.05). Diferencias más marcadas en Danceability (H = 155.9) y Energy (H = 119.7); la más débil, Valence (H = 11.4, p = 0.003). |
| **5 · Regresión** | R² = **0.154**: las métricas de audio explican poco las vistas. `Loudness` es el coeficiente más influyente (+0.70). |
| **6 · KNN** | F1 ponderado = **0.68** con k = 1. Buen desempeño en `album` (F1 0.77) y bajo en `compilation` (F1 0.13) y `single` (F1 0.28). |
| **7 · K-Means** | Silhouette = **0.24** con k = 4. Aparece un clúster claramente distinto: música muy instrumental, baja en energía y volumen (Instrumentalness ≈ 0.81). |
| **8 · Pronóstico** | R² = **0.34** (bajo poder predictivo). Tendencia decreciente de ≈ −3.5 M vistas por año; esto refleja que las canciones recientes han tenido menos tiempo para acumular vistas. |
| **PIA · Regresión** | R² = **0.958**, impulsado casi por completo por `Likes` (coef. 2.70). |
| **PIA · KNN** | F1 macro = **0.56** con k = 18 para predecir `official_video`. |
| **PIA · K-Means** | Silhouette = **0.79** con k = 5; los clústers separan canciones por nivel de popularidad (de ~24 M a ~6.4 mil M de vistas promedio). |

---

## ⚠️ Notas y limitaciones conocidas

- **Rutas relativas:** ejecuta siempre cada script desde su carpeta (ver [Cómo ejecutar](#️-cómo-ejecutar-los-scripts)).
- **Dos versiones del dataset limpio:** `Cleaning.py` genera un CSV de 9,218 filas, mientras que el archivo en `practica2/` (el que usa el resto del proyecto) tiene 8,638. Las prácticas 2–9 y el PIA trabajan con esta última versión.
- **Caracteres corruptos en textos:** algunos títulos y descripciones muestran símbolos como `ï¿½` heredados del CSV original.
- **`Cleaning.py` sobrescribe** `practica1/Songs_Dataset_Clean.csv` cada vez que se ejecuta.
- **Interpretación del PIA:** el R² de 0.958 se debe a que `Likes` está muy correlacionado con `Views`; conviene tomarlo como relación entre métricas de YouTube, no como capacidad de predecir popularidad desde el audio (eso es lo que mide la práctica 5, con R² = 0.15).
- **Práctica 8:** con solo ~13 años como puntos de datos y un R² bajo, el pronóstico es ilustrativo.
- `practica2.py`, `practica3.py`, `practica4.py` ejecutan su código al importarse (no usan `if __name__ == "__main__"`).

---

## 🛠 Solución de problemas

| Problema | Solución |
|---|---|
| `ModuleNotFoundError: No module named 'imblearn'` | `pip install imbalanced-learn` |
| `ModuleNotFoundError: No module named 'IPython'` | `pip install ipython` |
| `ModuleNotFoundError: No module named 'wordcloud'` | `pip install wordcloud` |
| `FileNotFoundError: ../practica2/Songs_Dataset_Clean.csv` | Estás ejecutando desde otra carpeta; haz `cd` a la carpeta de la práctica. |
| `graphviz.backend.execute.ExecutableNotFound` | Instala el programa Graphviz del sistema (ver [Instalación](#️-instalación)). |
| `UnicodeDecodeError` al leer el CSV crudo | Usa `encoding='latin1'` para `Songs_Dataset.csv`. |
| Las gráficas no aparecen (servidor/SSH) | Ejecuta con `MPLBACKEND=Agg` o usa Jupyter. |

---

## 👤 Autor

**noejsl** · [github.com/noejsl](https://github.com/noejsl)
