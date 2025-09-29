# --- 1. Importación de Librerías ---
# Se importan las herramientas necesarias para definir el DAG y ejecutar las tareas.
from airflow.decorators import dag, task  # Decoradores para definir DAGs y tareas de forma más sencilla (TaskFlow API).
from datetime import datetime
import pandas as pd
import numpy as np
import pickle
import os

# Librerías de Scikit-learn y XGBoost para el modelado.
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# --- 2. Definición de Rutas y Constantes ---
# Se definen rutas de salida para los artefactos que generará el pipeline (modelo, métricas, etc.).
# Usar os.path.join asegura que las rutas sean compatibles con cualquier sistema operativo.
BASE_DIR = os.path.dirname(__file__) # Directorio actual del DAG.
# La ruta sube dos niveles ('../../') para salir de 'dags/' y llegar al directorio raíz del proyecto.
MODEL_PATH = os.path.join(BASE_DIR, "../../models/trained_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "../../models/metrics/iris_metrics.csv")
PREDICTION_PATH = os.path.join(BASE_DIR, "../../models/metrics/sample_prediction.json")

# --- 3. Definición de Tareas ---
# Cada función decorada con @task se convierte en una tarea individual en Airflow.
# Airflow se encarga de pasar los resultados de una tarea a la siguiente a través de XComs.

@task
def split_data():
    """
    Tarea 1: Carga los datos de Iris y los divide en conjuntos de entrenamiento y prueba.
    Devuelve los datos divididos como un diccionario para que la siguiente tarea pueda usarlos.
    """
    print("--- Iniciando Tarea: split_data ---")
    iris = load_iris(as_frame=True)  # Carga el dataset como un DataFrame de pandas.
    X = iris.data
    y = iris.target
    
    # Divide los datos (80% entrenamiento, 20% prueba). random_state asegura la reproducibilidad.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Los DataFrames se convierten a diccionarios para ser pasados entre tareas vía XComs.
    return {
        "X_train": X_train.to_dict(),
        "X_test": X_test.to_dict(),
        "y_train": y_train.tolist(),
        "y_test": y_test.tolist()
    }


@task
def train_and_evaluate(data: dict):
    """
    Tarea 2: Recibe los datos, entrena múltiples modelos, evalúa su rendimiento,
    y guarda tanto las métricas como el mejor modelo.
    """
    print("--- Iniciando Tarea: train_and_evaluate ---")
    # Reconstruye los DataFrames a partir de los diccionarios recibidos de la tarea anterior.
    X_train = pd.DataFrame(data["X_train"])
    X_test = pd.DataFrame(data["X_test"])
    y_train = pd.Series(data["y_train"])
    y_test = pd.Series(data["y_test"])

    # Define los modelos a comparar.
    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "DecisionTree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=50, use_label_encoder=False, eval_metric="mlogloss", random_state=42
        ),
    }

    scores = {}
    # Itera sobre cada modelo para entrenarlo y evaluarlo.
    for name, model in models.items():
        model.fit(X_train, y_train)
        # .score() calcula la precisión (accuracy) por defecto para modelos de clasificación.
        acc = model.score(X_test, y_test)
        scores[name] = (acc, model) # Guarda la precisión y el objeto del modelo.
        print(f"🔹 {name} accuracy: {acc:.4f}")

    # Guarda las métricas de todos los modelos en un archivo CSV para auditoría.
    metrics_df = pd.DataFrame(
        [{"model": name, "accuracy": acc} for name, (acc, _) in scores.items()]
    )
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True) # Crea el directorio si no existe.
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"📊 Métricas guardadas en {METRICS_PATH}")

    # Selecciona el modelo con la mayor precisión.
    best_model_name = max(scores, key=lambda k: scores[k][0])
    best_model = scores[best_model_name][1]

    # Guarda el objeto del mejor modelo en un archivo .pkl usando pickle.
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    print(f"✅ Mejor modelo guardado: {best_model_name} en {MODEL_PATH}")
    # Devuelve el nombre del mejor modelo para que la siguiente tarea lo use.
    return best_model_name


@task
def test_prediction(best_model_name: str):
    """
    Tarea 3: Carga el modelo guardado y realiza una predicción de ejemplo
    para verificar que el artefacto funciona correctamente.
    """
    print("--- Iniciando Tarea: test_prediction ---")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Crea un dato de ejemplo (corresponde a una Iris Setosa).
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = int(model.predict(sample)[0])

    # Guarda el resultado de la predicción en un archivo JSON.
    result = {
        "best_model_used": best_model_name,
        "sample_input": sample.tolist(),
        "predicted_class": prediction,
        "predicted_species": "setosa" if prediction == 0 else "other"
    }

    os.makedirs(os.path.dirname(PREDICTION_PATH), exist_ok=True)
    pd.Series(result).to_json(PREDICTION_PATH, indent=4)
    print(f"🔮 Predicción de ejemplo guardada en {PREDICTION_PATH}")


# --- 4. Definición del DAG ---
# El decorador @dag agrupa las tareas en un pipeline.
@dag(
    schedule=None,                      # No se ejecuta automáticamente, solo bajo demanda (manual).
    start_date=datetime(2023, 1, 1),    # Fecha de inicio para la ejecución del DAG.
    catchup=False,                      # Evita que se ejecuten las instancias pasadas del DAG si Airflow estuvo detenido.
    dag_id="iris_pipeline_dag",         # Identificador único del DAG en la UI de Airflow.
    description="Pipeline completo: entrenamiento, métricas y predicción Iris",
    tags=["mlops", "iris", "didactico"], # Etiquetas para filtrar y organizar DAGs.
)
def iris_pipeline_dag():
    """
    Define el flujo de ejecución del pipeline.
    Airflow infiere las dependencias automáticamente:
    train_and_evaluate depende de split_data porque usa su salida (data).
    test_prediction depende de train_and_evaluate porque usa su salida (best_model_name).
    """
    data = split_data()
    best_model_name = train_and_evaluate(data)
    test_prediction(best_model_name)

# --- 5. Instanciación del DAG ---
# Esta línea crea el objeto DAG que Airflow buscará y cargará.
dag = iris_pipeline_dag()

# Mensaje de confirmación que aparece en los logs de Airflow al parsear el archivo.
print("✅ DAG iris_pipeline_dag cargado correctamente")