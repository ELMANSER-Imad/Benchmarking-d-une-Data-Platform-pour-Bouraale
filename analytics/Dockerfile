# Utiliser une image Ubuntu légère
FROM ubuntu:22.04

# Mettre à jour le système et installer les dépendances nécessaires
RUN apt-get update && apt-get install -y \
    python3 \
    python3-venv \
    curl \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# Créer un virtual environment Python
RUN python3 -m venv /opt/venv

# Mettre à jour pip et installer DuckDB et Polars
RUN /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install duckdb polars && \
    /opt/venv/bin/pip  install pandas  && \
    /opt/venv/bin/pip install requests



# Définir le répertoire de travail
WORKDIR /workspace

