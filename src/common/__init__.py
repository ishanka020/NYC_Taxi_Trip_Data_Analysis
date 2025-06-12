import os
from pathlib import Path
from pyspark.sql import SparkSession

def init_spark(app_name="NYC Taxi Analysis"):
    # Create SparkSession
    spark = SparkSession.builder \
        .appName(app_name) \
        .master("local[*]") \
        .getOrCreate()

    return spark

def get_project_paths():
    # Get root of the project
    project_root = Path(__file__).resolve().parents[2]

    # Define standard paths
    data_path = project_root / "data" / "clean" / "cleanedData1"
    output_path = project_root / "output"

    output_path.mkdir(parents=True, exist_ok=True)

    return data_path, output_path
