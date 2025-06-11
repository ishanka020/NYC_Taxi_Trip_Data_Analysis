from pyspark.sql import SparkSession
from pathlib import Path
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col
# Create SparkSession in local mode
spark = SparkSession.builder \
    .appName("NYC Taxi Data") \
    .master("local[*]") \
    .getOrCreate()

df = spark.read.parquet("/home/ariso/Documents/bigDataAssignment/NYC_Taxi_Trip_Data_Analysis/data/rawData")

df.printSchema()
df.show(5)
print("Total rows:", df.count())

required_cols = [
    "tpep_pickup_datetime", "tpep_dropoff_datetime",
    "fare_amount", "PULocationID", "DOLocationID",
    "trip_distance", "passenger_count"
]

df_cleaned = df.dropna(subset=required_cols)

print("Total rows:", df_cleaned.count())

df_cleaned.select("fare_amount").describe().show()#to see min max and get idea of outliers


df_cleaned = df_cleaned.filter(
    (col("fare_amount") > 3) & #the lowest base fare 
    (col("fare_amount") < 400)
)
df_cleaned.count()
df_cleaned.select("fare_amount").describe().show()

df_cleaned.select("trip_distance").describe().show()
df_cleaned = df_cleaned.filter(
  (col("trip_distance") > 0.62) & (col("trip_distance") < 120)
)
df_cleaned.count()
df_cleaned.select("trip_distance").describe().show()

df_cleaned.select("passenger_count").describe().show()
df_cleaned = df_cleaned.filter(
  (col("passenger_count") > 0) & (col("passenger_count") <= 4)
)
df_cleaned.count()
df_cleaned.select("passenger_count").describe().show()

df_cleaned = df_cleaned.filter(
  col("tpep_dropoff_datetime") > col("tpep_pickup_datetime")
)
df_cleaned.count()

df_cleaned.select("PULocationID").describe().show()
df_cleaned.select("DOLocationID").describe().show()

df_cleaned.select("tolls_amount").describe().show()
df_cleaned = df_cleaned.filter(
    (col("tolls_amount") >= 0) &
    (col("tolls_amount") < 100)
)
df_cleaned.count()

df_cleaned.select("tip_amount").describe().show()
df_cleaned = df_cleaned.filter(
    (col("tip_amount") >= 0) &
    (col("tip_amount") < 50)
)

df_cleaned = df_cleaned.filter(
    (col("total_amount") > 0) &
    (col("total_amount") < 500)
)
df_cleaned.count()

# Save cleaned data
df_cleaned.write.mode("overwrite").parquet("/home/ariso/Documents/bigDataAssignment/NYC_Taxi_Trip_Data_Analysis/data/cleanedData")

# Stop Spark session
spark.stop()