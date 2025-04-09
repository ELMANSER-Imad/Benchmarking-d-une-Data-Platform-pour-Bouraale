import os
import dask.dataframe as dd
from dask.distributed import Client
import time
import duckdb
from pyspark.sql import SparkSession

# Configuration
INPUT_FILE = "data/measurements.txt"
OUTPUT_DIR = "data/formats"
PARTITIONS = 4  # Nombre de partitions pour le traitement distribué

def ingest_with_dask():
    try:
        # Initialiser le client Dask
        client = Client(n_workers=PARTITIONS)
        print(f"Dask Dashboard: {client.dashboard_link}")

        # Lecture du fichier texte
        print("Lecture des données avec Dask...")
        start_time = time.time()
        
        df = dd.read_csv(
            INPUT_FILE,
            sep=";",
            header=None,
            names=["station", "measurement"],
            dtype={"station": "str", "measurement": "float64"}
        )
        
        # Persister les données en mémoire pour les opérations suivantes
        df = df.persist()
        print(f"Temps de lecture: {time.time() - start_time:.2f}s")
        print(f"Nombre de partitions: {df.npartitions}")
        print(f"Nombre total de lignes: {len(df):,}")

        # Créer le répertoire de sortie
        os.makedirs(OUTPUT_DIR, exist_ok=True)

        # Sauvegarde en Parquet
        parquet_path = os.path.join(OUTPUT_DIR, "measurements_dask.parquet")
        start_time = time.time()
        df.to_parquet(parquet_path, engine="pyarrow")
        print(f"Parquet sauvegardé en {time.time() - start_time:.2f}s")

        # Sauvegarde en CSV (déconseillé pour les gros volumes)
        csv_path = os.path.join(OUTPUT_DIR, "measurements_dask.csv")
        start_time = time.time()
        df.to_csv(csv_path, single_file=True)
        print(f"CSV sauvegardé en {time.time() - start_time:.2f}s")

        # Conversion en Pandas DataFrame pour les autres formats
        # Note: Ceci n'est recommandé que si les données tiennent en mémoire
        pd_df = df.compute()

        # Sauvegarde en ORC via DuckDB
        orc_path = os.path.join(OUTPUT_DIR, "measurements_dask.orc")
        start_time = time.time()
        con = duckdb.connect()
        con.execute(f"COPY (SELECT * FROM pd_df) TO '{orc_path}' (FORMAT 'orc')")
        print(f"ORC sauvegardé en {time.time() - start_time:.2f}s")

        # Sauvegarde en Delta Lake et Iceberg via Spark
        spark = SparkSession.builder \
            .appName("Dask Ingestion") \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
            .getOrCreate()

        # Convertir en Spark DataFrame
        spark_df = spark.createDataFrame(pd_df)

        # Delta Lake
        delta_path = os.path.join(OUTPUT_DIR, "measurements_dask.delta")
        start_time = time.time()
        spark_df.write.format("delta").mode("overwrite").save(delta_path)
        print(f"Delta Lake sauvegardé en {time.time() - start_time:.2f}s")

        # Iceberg
        iceberg_path = os.path.join(OUTPUT_DIR, "measurements_dask.iceberg")
        start_time = time.time()
        spark_df.write.format("iceberg").mode("overwrite").save(iceberg_path)
        print(f"Iceberg sauvegardé en {time.time() - start_time:.2f}s")

    except Exception as e:
        print(f"Erreur lors de l'ingestion: {e}")
        raise
    finally:
        client.close()

if __name__ == "__main__":
    ingest_with_dask()