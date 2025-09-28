from airflow.decorators import dag, task
from datetime import datetime
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import pickle
import os
import pandas as pd
import numpy as np

# Rutas de salida
BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "../../models/trained_model.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "../../models/metrics/iris_metrics.csv")
PREDICTION_PATH = os.path.join(BASE_DIR, "../../models/metrics/sample_prediction.json")

@task
def split_data():
    iris = load_iris(as_frame=True)
    X = iris.data.to_dict()
    y = iris.target.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        pd.DataFrame(X), pd.Series(y), test_size=0.2, random_state=42
    )
    return {
        "X_train": X_train.to_dict(),
        "X_test": X_test.to_dict(),
        "y_train": y_train.tolist(),
        "y_test": y_test.tolist()
    }


@task
def train_and_evaluate(data: dict):
    """Entrena varios modelos, evalúa y guarda métricas + mejor modelo."""
    X_train = pd.DataFrame(data["X_train"])
    X_test = pd.DataFrame(data["X_test"])
    y_train = pd.Series(data["y_train"])
    y_test = pd.Series(data["y_test"])



    models = {
        "LogisticRegression": LogisticRegression(max_iter=200),
        "DecisionTree": DecisionTreeClassifier(max_depth=3, random_state=42),
        "XGBoost": XGBClassifier(
            n_estimators=50, use_label_encoder=False, eval_metric="mlogloss", random_state=42
        ),
    }

    scores = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)
        scores[name] = (acc, model)
        print(f"🔹 {name} accuracy: {acc:.4f}")

    # Guardar métricas en CSV
    metrics_df = pd.DataFrame(
        [{"model": name, "accuracy": acc} for name, (acc, _) in scores.items()]
    )
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    metrics_df.to_csv(METRICS_PATH, index=False)
    print(f"📊 Métricas guardadas en {METRICS_PATH}")

    # Seleccionar mejor modelo
    best_model_name = max(scores, key=lambda k: scores[k][0])
    best_model = scores[best_model_name][1]

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)

    print(f"✅ Mejor modelo guardado: {best_model_name} en {MODEL_PATH}")
    return best_model_name


@task
def test_prediction(best_model_name: str):
    """Carga el modelo guardado y genera una predicción de ejemplo."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    # Ejemplo de flor Iris
    sample = np.array([[5.1, 3.5, 1.4, 0.2]])
    prediction = int(model.predict(sample)[0])

    result = {
        "best_model": best_model_name,
        "sample_input": sample.tolist(),
        "predicted_class": prediction
    }

    os.makedirs(os.path.dirname(PREDICTION_PATH), exist_ok=True)
    pd.Series(result).to_json(PREDICTION_PATH)
    print(f"🔮 Predicción de ejemplo guardada en {PREDICTION_PATH}")


@dag(
    schedule=None,
    start_date=datetime(2023, 1, 1),
    catchup=False,
    dag_id="iris_pipeline_dag",
    description="Pipeline completo: entrenamiento, métricas y predicción Iris",
    tags=["mlops", "iris", "didactico"],
)
def iris_pipeline_dag():
    """Pipeline completo: split → entrenamiento → métricas + modelo → predicción."""
    data = split_data()
    best_model_name = train_and_evaluate(data)
    test_prediction(best_model_name)


dag = iris_pipeline_dag()

print("✅ DAG iris_pipeline_dag cargado correctamente")
