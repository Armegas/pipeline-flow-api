from airflow.decorators import dag, task
from datetime import datetime

@task
def hello():
    print("Hola desde el DAG minimal")

@dag(
    dag_id="iris_minimal_dag",
    start_date=datetime(2023, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=["debug"]
)
def iris_minimal_dag():
    hello()

dag = iris_minimal_dag()