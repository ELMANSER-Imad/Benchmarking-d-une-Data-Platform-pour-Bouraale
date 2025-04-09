import time
import os
import duckdb
import polars as pl
import dask.dataframe as dd
from pyspark.sql import SparkSession
from dask.distributed import Client
import matplotlib.pyplot as plt
import pandas as pd

# Configuration
DATA_DIR = "data/formats"
RESULTS_DIR = "results"
SCALE_FACTORS = ["SF10", "SF50", "SF100"]  # À adapter selon vos données
FORMATS = ["parquet", "orc", "delta", "iceberg"]
TECHNOLOGIES = ["polars", "duckdb", "dask", "spark"]

# Requêtes TPC-H adaptées pour 1BRC
QUERIES = {
    "q1": """
        SELECT 
            station,
            COUNT(*) as count,
            AVG(measurement) as avg_temp,
            MIN(measurement) as min_temp,
            MAX(measurement) as max_temp
        FROM measurements
        GROUP BY station
        ORDER BY station
    """,
    "q2": """
        SELECT 
            station,
            measurement,
            RANK() OVER (PARTITION BY station ORDER BY measurement DESC) as rank
        FROM measurements
        WHERE measurement > (SELECT AVG(measurement) FROM measurements)
    """
}

def run_benchmark():
    results = []
    
    # Créer le répertoire des résultats
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Initialiser Spark
    spark = SparkSession.builder \
        .appName("TPC-H Benchmark") \
        .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
        .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
        .getOrCreate()
    
    # Initialiser Dask
    dask_client = Client()
    
    for sf in SCALE_FACTORS:
        data_path = os.path.join(DATA_DIR, f"measurements_{sf}")
        
        for file_format in FORMATS:
            file_path = f"{data_path}.{file_format}"
            
            if not os.path.exists(file_path):
                print(f"Fichier {file_path} non trouvé, ignoré...")
                continue
                
            print(f"\nBenchmarking {file_format.upper()} avec {sf}...")
            
            # Benchmark Polars
            if "polars" in TECHNOLOGIES:
                try:
                    start = time.time()
                    df = pl.read_parquet(file_path) if file_format == "parquet" else pl.read_ipc(file_path)
                    df.query(QUERIES["q1"])
                    duration = time.time() - start
                    results.append({
                        "tech": "polars",
                        "format": file_format,
                        "sf": sf,
                        "query": "q1",
                        "time": duration
                    })
                    print(f"Polars Q1: {duration:.2f}s")
                except Exception as e:
                    print(f"Erreur avec Polars: {e}")
            
            # Benchmark DuckDB
            if "duckdb" in TECHNOLOGIES:
                try:
                    con = duckdb.connect()
                    start = time.time()
                    con.execute(f"CREATE TABLE measurements AS SELECT * FROM '{file_path}'")
                    con.execute(QUERIES["q1"])
                    duration = time.time() - start
                    results.append({
                        "tech": "duckdb",
                        "format": file_format,
                        "sf": sf,
                        "query": "q1",
                        "time": duration
                    })
                    print(f"DuckDB Q1: {duration:.2f}s")
                    con.close()
                except Exception as e:
                    print(f"Erreur avec DuckDB: {e}")
            
            # Benchmark Dask
            if "dask" in TECHNOLOGIES:
                try:
                    start = time.time()
                    df = dd.read_parquet(file_path) if file_format == "parquet" else dd.read_orc(file_path)
                    result = df.groupby("station").agg(["count", "mean", "min", "max"]).compute()
                    duration = time.time() - start
                    results.append({
                        "tech": "dask",
                        "format": file_format,
                        "sf": sf,
                        "query": "q1",
                        "time": duration
                    })
                    print(f"Dask Q1: {duration:.2f}s")
                except Exception as e:
                    print(f"Erreur avec Dask: {e}")
            
            # Benchmark Spark
            if "spark" in TECHNOLOGIES:
                try:
                    start = time.time()
                    df = spark.read.format(file_format).load(file_path)
                    df.createOrReplaceTempView("measurements")
                    spark.sql(QUERIES["q1"]).count()  # Force l'exécution
                    duration = time.time() - start
                    results.append({
                        "tech": "spark",
                        "format": file_format,
                        "sf": sf,
                        "query": "q1",
                        "time": duration
                    })
                    print(f"Spark Q1: {duration:.2f}s")
                except Exception as e:
                    print(f"Erreur avec Spark: {e}")
    
    # Sauvegarder les résultats
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "benchmark_results.csv"), index=False)
    
    # Générer des visualisations
    generate_visualizations(results_df)
    
    # Nettoyage
    dask_client.close()
    spark.stop()

def generate_visualizations(results_df):
    # Graphique par technologie
    for tech in TECHNOLOGIES:
        tech_df = results_df[results_df["tech"] == tech]
        if not tech_df.empty:
            plt.figure(figsize=(10, 6))
            for fmt in tech_df["format"].unique():
                fmt_df = tech_df[tech_df["format"] == fmt]
                plt.plot(fmt_df["sf"], fmt_df["time"], label=fmt, marker="o")
            
            plt.title(f"Performance de {tech.capitalize()} par format et échelle")
            plt.xlabel("Scale Factor")
            plt.ylabel("Temps d'exécution (s)")
            plt.legend()
            plt.grid()
            plt.savefig(os.path.join(RESULTS_DIR, f"performance_{tech}.png"))
            plt.close()
    
    # Graphique comparatif global
    plt.figure(figsize=(12, 8))
    for tech in TECHNOLOGIES:
        tech_df = results_df[results_df["tech"] == tech]
        if not tech_df.empty:
            avg_times = tech_df.groupby("sf")["time"].mean()
            plt.plot(avg_times.index, avg_times.values, label=tech, marker="o", linestyle="--")
    
    plt.title("Comparaison globale des technologies")
    plt.xlabel("Scale Factor")
    plt.ylabel("Temps moyen d'exécution (s)")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(RESULTS_DIR, "comparaison_globale.png"))
    plt.close()

if __name__ == "__main__":
    run_benchmark()