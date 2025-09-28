from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import os

# Ruta absoluta del modelo entrenado
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "trained_model.pkl")

# Cargar modelo con manejo de errores
try:
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    raise RuntimeError(f"No se encontró el modelo en {MODEL_PATH}")
except Exception as e:
    raise RuntimeError(f"Error al cargar el modelo: {e}")

# Inicializar API
app = FastAPI(title="Iris Prediction API")

# Esquema de entrada
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Mapeo de clases
SPECIES_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# Ruta raíz (healthcheck)
@app.get("/")
def root():
    return {"status": "API funcionando", "model_path": MODEL_PATH}

# Endpoint de predicción
@app.post("/predict")
def predict_species(data: IrisInput):
    try:
        input_array = np.array([[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]])
        prediction = int(model.predict(input_array)[0])
        species = SPECIES_MAP.get(prediction, "desconocida")
        return {
            "predicted_class": prediction,
            "predicted_species": species
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error en la predicción: {e}")
