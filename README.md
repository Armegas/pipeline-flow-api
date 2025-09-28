
# pipeline-flow-api
📦 pipeline-flow-api
🚀 Descripción
pipeline-flow-api es un proyecto educativo y modular que integra Airflow para orquestación, FastAPI para servir modelos y endpoints, y prácticas de MLOps reproducibles. Su objetivo es mostrar el camino del notebook a producción, con un enfoque didáctico y auditable.

📂 Ejemplo para tu estructura de proyecto
En tu README.md ponlo así:

<pre>

bash
pipeline-flow-api/
├── api/              # Endpoints FastAPI
├── dags/             # DAGs de Airflow
├── models/           # Modelos entrenados (ignorado en Git)
├── notebooks/        # Experimentos y prototipos
├── utils/            # Funciones auxiliares
├── main.py           # Entry point FastAPI
├── requirements.txt  # Dependencias
└── README.md
</pre>

<pre>

⚙️ Instalación y entorno
bash
# Crear entorno virtual
python3 -m venv airflow-env
source airflow-env/bin/activate

# Instalar dependencias
pip install -r requirements.txt
</pre>

▶️ Uso
Levantar Airflow
<pre>
bash
airflow standalone
</pre>
DAGs disponibles en </pre> dags/. </pre>

Levantar FastAPI
<pre>
bash
uvicorn main:app --reload
</pre>
API disponible en: http://localhost:8000/docs

🧪 Tests
<pre>
bash
pytest
</pre>

📖 Roadmap
<pre>
[x] Plantilla reproducible de MLOps

[ ] Integración con Prefect como alternativa a Airflow

[ ] Dashboard de métricas en tiempo real
</pre>

🤝 Contribución
<pre>
Haz un fork del repo

Crea una rama (git checkout -b feature/nueva-funcionalidad)

Haz commit de tus cambios

Haz push a la rama

Abre un Pull Request
</pre>


<pre>
📜 Licencia
MIT License.
</pre>
