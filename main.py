# --- 1. Importación de Librerías ---
# Se importan las herramientas necesarias para crear la API y procesar los datos.
from fastapi import FastAPI, HTTPException  # FastAPI para crear la API, HTTPException para manejar errores HTTP.
from pydantic import BaseModel            # Pydantic para la validación de datos de entrada y salida.
import pickle                             # Para deserializar (cargar) el modelo entrenado.
import numpy as np                        # Para crear arrays numéricos que el modelo pueda procesar.
import os                                 # Para interactuar con el sistema de archivos y construir rutas.

# --- 2. Configuración y Carga del Modelo ---
# Se define la ruta al modelo entrenado de forma robusta para que funcione
# independientemente desde dónde se ejecute el script.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# La ruta se ajusta para subir un nivel ('..') y encontrar la carpeta 'models'.
# Esto asume una estructura de proyecto: /api/main.py y /models/trained_model.pkl
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "trained_model.pkl")

# Se carga el modelo serializado (.pkl) en memoria al iniciar la API.
# El manejo de errores asegura que la aplicación no se inicie si el modelo no se encuentra.
try:
    # Se abre el archivo en modo lectura binaria ("rb").
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
except FileNotFoundError:
    # Si el archivo del modelo no existe, se lanza un error claro que detiene la app.
    raise RuntimeError(f"No se encontró el modelo en la ruta: {MODEL_PATH}")
except Exception as e:
    # Captura cualquier otro error durante la carga (ej. archivo corrupto, error de pickle).
    raise RuntimeError(f"Error al cargar el modelo: {e}")

# --- 3. Inicialización de la API ---
# Se crea una instancia de FastAPI. El título, descripción y versión se mostrarán
# en la documentación automática (en /docs y /redoc).
app = FastAPI(
    title="Iris Prediction API",
    description="Una API para predecir la especie de la flor Iris usando un modelo de ML.",
    version="1.0.0"
)

# --- 4. Definición de Esquemas de Datos (Pydantic) ---
# Pydantic se usa para validar los tipos de datos de entrada y salida.
# Esto previene errores y genera documentación automática muy clara.

class IrisInput(BaseModel):
    """Define la estructura y tipos de datos esperados en el cuerpo (body) de la petición POST."""
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    class Config:
        """Clase de configuración interna para el modelo Pydantic."""
        # Proporciona un ejemplo que se mostrará en la documentación de la API.
        # Esto facilita enormemente las pruebas desde la interfaz de /docs.
        schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

class PredictionOutput(BaseModel):
    """Define la estructura y tipos de datos de la respuesta que enviará la API."""
    predicted_class: int
    predicted_species: str

# --- 5. Lógica de la Aplicación ---
# Mapeo para convertir la predicción numérica del modelo (0, 1, 2) a un nombre legible por humanos.
SPECIES_MAP = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

# --- 6. Definición de Endpoints de la API ---

@app.get("/", tags=["Health Check"])
def root():
    """
    Endpoint raíz o de 'health check'.
    Permite verificar rápidamente que la API está en funcionamiento.
    Es una buena práctica tener un endpoint simple como este.
    El argumento 'tags' agrupa este endpoint en la documentación.
    """
    return {"status": "API funcionando correctamente"}

@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
def predict_species(data: IrisInput) -> PredictionOutput:
    """
    Endpoint para realizar predicciones.
    - Recibe las 4 características de una flor Iris en formato JSON (validadas por IrisInput).
    - Devuelve la especie predicha por el modelo (validada por PredictionOutput).
    - `response_model=PredictionOutput` asegura que la respuesta siempre siga ese esquema.
    """
    try:
        # 1. Convierte los datos de entrada (Pydantic) a un array de NumPy.
        #    El modelo de scikit-learn espera un array 2D (una lista de muestras),
        #    por eso el doble corchete [[...]] para crear una matriz de 1x4.
        input_array = np.array([[
            data.sepal_length,
            data.sepal_width,
            data.petal_length,
            data.petal_width
        ]])

        # 2. Realiza la predicción usando el modelo cargado.
        #    .predict() devuelve un array, por eso se accede al primer elemento [0].
        prediction_raw = model.predict(input_array)
        prediction = int(prediction_raw[0])

        # 3. Mapea el resultado numérico a su nombre de especie.
        #    .get() es más seguro que [] porque devuelve un valor por defecto si la clave no existe.
        species = SPECIES_MAP.get(prediction, "Especie desconocida")

        # 4. Devuelve el resultado estructurado. FastAPI se encarga de convertir este
        #    objeto Pydantic a una respuesta JSON.
        return PredictionOutput(
            predicted_class=prediction,
            predicted_species=species
        )
    except Exception as e:
        # Si ocurre cualquier error durante la predicción, se devuelve un error HTTP 500.
        # Esto evita que la API se caiga y da una respuesta controlada al cliente.
        raise HTTPException(status_code=500, detail=f"Error en el proceso de predicción: {e}")

