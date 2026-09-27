# =============================================================================
# Airflow image with the pipeline's dependencies
# DDM501 - Lab 2
# =============================================================================
FROM apache/airflow:2.8.4-python3.11

USER airflow

COPY requirements-airflow.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-airflow.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.8.4/constraints-3.11.txt"

ENV PYTHONPATH="/opt/airflow/project:${PYTHONPATH}"