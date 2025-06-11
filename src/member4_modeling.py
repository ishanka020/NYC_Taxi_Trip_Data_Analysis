from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, hour, dayofweek, unix_timestamp, when, log1p, sqrt, skewness,
    month, monotonically_increasing_id
)
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import (
    LinearRegression, RandomForestRegressor, GBTRegressor
)
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import RegressionEvaluator
from pathlib import Path

# Initialize Spark
spark = SparkSession.builder \
    .appName("FareTipModeling") \
    .config("spark.driver.memory", "4g") \
    .master("local[*]") \
    .getOrCreate()

# Load cleaned dataset
# Set folder path (raw string to avoid escape errors)
folder_path = Path(r"D:\L4S2\Big Data\assignment\NYC_Taxi_Trip_Data_Analysis\data\cleaned")

# Get list of all .parquet files
parquet_files = [str(p) for p in folder_path.glob("*.parquet")]

# Read all parquet files into a single DataFrame
df = spark.read.parquet(*parquet_files)
# =========================
# Feature Engineering
# =========================

# Convert to timestamp if needed
df = df.withColumn("pickup_datetime", col("tpep_pickup_datetime").cast("timestamp"))
df = df.withColumn("dropoff_datetime", col("tpep_dropoff_datetime").cast("timestamp"))

# Time features
df = df.withColumn("trip_duration_minutes",
                   (unix_timestamp("dropoff_datetime") - unix_timestamp("pickup_datetime")) / 60)
df = df.withColumn("pickup_hour", hour("pickup_datetime"))
df = df.withColumn("pickup_dayofweek", dayofweek("pickup_datetime"))
df = df.withColumn("pickup_month", month("pickup_datetime"))
df = df.withColumn("is_weekend", when(col("pickup_dayofweek") >= 6, 1).otherwise(0))
df = df.withColumn("is_night", when((col("pickup_hour") < 6) | (col("pickup_hour") >= 22), 1).otherwise(0))

# Derived features
df = df.withColumn("tip_per_mile", when(col("trip_distance") != 0, col("tip_amount") / col("trip_distance")).otherwise(0))
df = df.withColumn("tip_percent_fare", when(col("fare_amount") != 0, col("tip_amount") / col("fare_amount")).otherwise(0))
df = df.withColumn("fare_per_minute", when(col("trip_duration_minutes") != 0, col("fare_amount") / col("trip_duration_minutes")).otherwise(0))

# Airport and zone tags
airport_ids = [1, 132, 138, 140]
df = df.withColumn("is_airport_trip", when(col("PULocationID").isin(airport_ids) | col("DOLocationID").isin(airport_ids), 1).otherwise(0))
df = df.withColumn("is_same_zone", when(col("PULocationID") == col("DOLocationID"), 1).otherwise(0))

# Filter unrealistic records
df = df.filter((col("trip_distance") > 0) & (col("trip_distance") < 100))
df = df.filter((col("fare_amount") > 0) & (col("fare_amount") < 200))
df = df.filter((col("tip_amount") >= 0) & (col("tip_amount") < 100))

# =========================
# Transformations
# =========================
log_cols = ["trip_distance", "fare_amount", "tolls_amount", "Airport_fee", "passenger_count"]
for c in log_cols:
    df = df.withColumn(f"log_{c}", log1p(col(c)))
df = df.withColumn("sqrt_log_trip_distance", sqrt(col("log_trip_distance")))

# =========================
# Feature Assembly
# =========================
final_features = [
    "sqrt_log_trip_distance", "log_fare_amount", "log_tolls_amount",
    "log_Airport_fee", "log_passenger_count", "trip_duration_minutes",
    "pickup_hour", "pickup_dayofweek", "is_weekend", "is_night",
    "tip_per_mile", "tip_percent_fare", "fare_per_minute",
    "is_airport_trip", "is_same_zone"
]

assembler = VectorAssembler(inputCols=final_features, outputCol="features")
df = assembler.transform(df).select("features", "tip_amount")

# Train-test split
train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)

# =========================
# Models
# =========================

# --- Linear Regression ---
lr = LinearRegression(featuresCol="features", labelCol="tip_amount")
lr_model = lr.fit(train_data)
lr_preds = lr_model.transform(test_data)

# --- Random Forest ---
rf = RandomForestRegressor(featuresCol="features", labelCol="tip_amount", seed=42)
rf_model = rf.fit(train_data)
rf_preds = rf_model.transform(test_data)

# --- Gradient Boosted Trees ---
gbt = GBTRegressor(featuresCol="features", labelCol="tip_amount", maxIter=100)
gbt_model = gbt.fit(train_data)
gbt_preds = gbt_model.transform(test_data)

# --- Hyperparameter Tuning for GBT ---
param_grid = ParamGridBuilder() \
    .addGrid(gbt.maxDepth, [3, 5]) \
    .addGrid(gbt.maxIter, [20, 50]) \
    .addGrid(gbt.stepSize, [0.1]) \
    .build()

tvs = TrainValidationSplit(
    estimator=gbt,
    estimatorParamMaps=param_grid,
    evaluator=RegressionEvaluator(labelCol="tip_amount", predictionCol="prediction", metricName="rmse"),
    trainRatio=0.8
)
tuned_gbt_model = tvs.fit(train_data)
tuned_preds = tuned_gbt_model.transform(test_data)

# =========================
# Model Blending
# =========================
test_data_id = test_data.withColumn("row_id", monotonically_increasing_id())

rf_blend = rf_model.transform(test_data_id).select("row_id", "prediction").withColumnRenamed("prediction", "rf_pred")
gbt_blend = gbt_model.transform(test_data_id).select("row_id", "prediction").withColumnRenamed("prediction", "gbt_pred")

blended = rf_blend.join(gbt_blend, on="row_id")
blended = blended.withColumn("final_prediction", (col("rf_pred") + col("gbt_pred")) / 2)

true_labels = test_data_id.select("row_id", "tip_amount")
final_eval = blended.join(true_labels, on="row_id")

# =========================
# Evaluation
# =========================
evaluator_rmse = RegressionEvaluator(labelCol="tip_amount", predictionCol="prediction", metricName="rmse")
evaluator_r2 = RegressionEvaluator(labelCol="tip_amount", predictionCol="prediction", metricName="r2")

print(f"Linear Regression RMSE: {evaluator_rmse.evaluate(lr_preds):.3f}")
print(f"Random Forest RMSE: {evaluator_rmse.evaluate(rf_preds):.3f}")
print(f"GBT RMSE: {evaluator_rmse.evaluate(gbt_preds):.3f}")
print(f"Tuned GBT RMSE: {evaluator_rmse.evaluate(tuned_preds):.3f}")
print(f"Blended Model RMSE: {RegressionEvaluator(labelCol='tip_amount', predictionCol='final_prediction', metricName='rmse').evaluate(final_eval):.3f}")
print(f"Blended Model R2: {RegressionEvaluator(labelCol='tip_amount', predictionCol='final_prediction', metricName='r2').evaluate(final_eval):.3f}")
