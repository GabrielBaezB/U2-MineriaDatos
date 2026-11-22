
# Evaluación Sumativa Unidad 2 — Minería de Datos (`IF308IINF`)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/GabrielBaezB/U2-MineriaDatos/blob/main/Evaluacion_U2_Mineria_de_Datos.ipynb)

Desarrollo integral en Python (Jupyter Notebook) de la **Evaluación Sumativa N° 2** de la asignatura **Minería de Datos** — **Instituto Profesional San Sebastián (IPSS)**, carrera **Ingeniería en Informática**.

* **Estudiante:** Gabriel Ignacio Báez Báez  
* **Docente:** Nelson Carvallo Suma de Villa  
* **Sección:** `T3-26-(51)`

---

## Descripción General y Enfoque Metodológico

El notebook [`Evaluacion_U2_Mineria_de_Datos.ipynb`](Evaluacion_U2_Mineria_de_Datos.ipynb) resuelve los tres desafíos clásicos de aprendizaje automático propuestos en la pauta (clasificación supervisada, clustering no supervisado y regresión predictiva), incorporando además un flujo de **Nivel Sobresaliente**: preprocesamiento reproducible (`SimpleImputer`, `StandardScaler`, `Pipeline`), reducción de dimensionalidad (`PCA`), ajuste de hiperparámetros con validación cruzada (`GridSearchCV`, `StratifiedKFold`, `KFold`), comparación frente a algoritmos alternativos y evaluación crítica de métricas y limitaciones.

| Desafío | Dataset | Paradigma | Modelo Base (Pauta) | Optimización y Modelos Comparativos | Métricas Aplicadas |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Clasificación** | **Iris Dataset** (`150 × 4`) | Supervisado (Multiclase) | `KNN (k = 3)` | `Pipeline + StandardScaler`, `GridSearchCV`, `PCA 2D`, `Random Forest`, `SVM (RBF)` | `Accuracy`, `Precision`, `Recall`, `F1-Score`, Matriz de Confusión, `CV-5` |
| **2. Clustering** | **Mall Customers** (`200 × 5`) | No Supervisado | `K-Means (k = 2, 3)` | `StandardScaler`, Método del Codo, Silueta (`k = 5`), Jerárquico Ward (Dendrograma), `DBSCAN`, extensión 3D + `PCA 2D` | Inercia (`WCSS`), `Silhouette Score`, `Davies-Bouldin`, Perfilamiento 2D |
| **3. Regresión** | **Boston Housing** (`506 × 14`) | Supervisado (Continuo) | `LinearRegression` simple (`RM`) | `SimpleImputer`, Regresión Múltiple, `RidgeCV`, `LassoCV`, `RandomForestRegressor` (`GridSearchCV`) | $R^2$, $R^2$ ajustado, `MAE`, `MSE`, `RMSE`, Análisis de Residuos, Feature Importance |

---

## Estructura del Repositorio

```text
├── Evaluacion_U2_Mineria_de_Datos.ipynb   # Notebook completo (Desafíos 1 al 5)
├── Mall_Customers.csv                     # Dataset de segmentación de clientes (200 × 5)
├── BostonHousing.csv                      # Dataset inmobiliario de Boston (506 × 14)
├── requirements.txt                       # Dependencias Python del proyecto
├── .gitignore                             # Exclusión de informes privados y temporales
└── README.md                              # Documentación del proyecto
```

> **Nota de reproducibilidad:** El dataset *Iris* se carga nativamente desde `sklearn.datasets.load_iris()`. Los archivos `Mall_Customers.csv` y `BostonHousing.csv` se leen de forma local y, en caso de ejecutarse de manera aislada en Google Colab, el notebook incluye carga automática de respaldo mediante URL pública. Asimismo, se evita `load_boston()` (removido en `scikit-learn >= 1.2` por consideraciones éticas) y se utiliza la versión CSV equivalente documentada en las referencias.

---

## Ejecución

### Opción A: Google Colab (Recomendado)
Haz clic en el botón **Open in Colab** al inicio de este README y ejecuta **Entorno de ejecución → Ejecutar todas**.

### Opción B: Entorno Local
```bash
pip install -r requirements.txt
jupyter notebook Evaluacion_U2_Mineria_de_Datos.ipynb
```
