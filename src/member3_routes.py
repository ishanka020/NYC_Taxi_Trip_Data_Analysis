from pyspark.sql import SparkSession
from pathlib import Path
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.sql.functions import col, hour, dayofweek, dayofmonth, avg, count, sum as spark_sum, desc, asc
from pyspark.sql.functions import round as spark_round, when, isnan, isnull
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType
import pandas as pd

# Create SparkSession in local mode
spark = SparkSession.builder \
    .appName("FareTipModeling") \
    .master("local[*]") \
    .getOrCreate()

folder_path = Path(r"\Your-path\data\cleanedData")
parquet_files = sorted(folder_path.glob("*.parquet"))

print(f"Found {len(parquet_files)} parquet files:")
all_dfs = []
for f in parquet_files:
    print(f"\nReading: {f.name}")
    df = spark.read.parquet(str(f))
    all_dfs.append(df)
    df.show(5)

    # Combine all DataFrames
if len(all_dfs) > 1:
    combined_df = all_dfs[0]
    for df in all_dfs[1:]:
        combined_df = combined_df.union(df)
else:
    combined_df = all_dfs[0]

print(f"\nCombined dataset shape: {combined_df.count()} rows, {len(combined_df.columns)} columns")
print("\nCombined dataset schema:")
combined_df.printSchema()
# MEMBER 3: GEOSPATIAL AND ROUTE ANALYSIS
# ========================================

print("\n" + "="*60)
print("MEMBER 3: GEOSPATIAL AND ROUTE ANALYSIS")
print("="*60)

# Step 1: Data Quality Check for Location IDs
print("\n1. DATA QUALITY CHECK FOR LOCATION IDs")
print("-" * 40)
# Check for null/invalid location IDs
null_pickup = combined_df.filter(col("PULocationID").isNull() | (col("PULocationID") == 0)).count()
null_dropoff = combined_df.filter(col("DOLocationID").isNull() | (col("DOLocationID") == 0)).count()

print(f"Records with null/zero pickup location: {null_pickup}")
print(f"Records with null/zero dropoff location: {null_dropoff}")

# Filter out invalid location records
valid_trips = combined_df.filter(
    col("PULocationID").isNotNull() & 
    col("DOLocationID").isNotNull() & 
    (col("PULocationID") > 0) & 
    (col("DOLocationID") > 0)
)

print(f"Valid trips after filtering: {valid_trips.count()}")
# Step 2: Zone Analysis - Most Popular Pickup and Dropoff Zones
print("\n2. ZONE ANALYSIS - PICKUP AND DROPOFF PATTERNS")
print("-" * 50)
# Most popular pickup zones
print("Top 15 Pickup Zones:")
pickup_zones = valid_trips.groupBy("PULocationID") \
    .agg(count("*").alias("pickup_count")) \
    .orderBy(desc("pickup_count"))

pickup_zones.show(15)
# Most popular dropoff zones
print("\nTop 15 Dropoff Zones:")
dropoff_zones = valid_trips.groupBy("DOLocationID") \
    .agg(count("*").alias("dropoff_count")) \
    .orderBy(desc("dropoff_count"))

dropoff_zones.show(15)

# Step 3: Route Analysis - Most Common Routes
print("\n3. ROUTE ANALYSIS - MOST COMMON ROUTES")
print("-" * 45)
# First, let's check what columns are available
print("Available columns in the dataset:")
print(combined_df.columns)

# Calculate trip duration from pickup and dropoff times
from pyspark.sql.functions import unix_timestamp, to_timestamp

# Add trip duration calculation
trips_with_duration = valid_trips.withColumn(
    "trip_duration_minutes",
    (unix_timestamp("tpep_dropoff_datetime") - unix_timestamp("tpep_pickup_datetime")) / 60
)

 # Analyze most common pickup-dropoff zone pairs
routes_analysis = trips_with_duration.groupBy("PULocationID", "DOLocationID") \
    .agg(
        count("*").alias("trip_count"),
        avg("trip_distance").alias("avg_distance"),
        avg("trip_duration_minutes").alias("avg_duration_minutes"),
        avg("fare_amount").alias("avg_fare")
    ) \
    .filter(col("trip_count") >= 10) \
    .orderBy(desc("trip_count"))

print("Top 20 Most Common Routes (Pickup -> Dropoff):")
routes_analysis.select(
    col("PULocationID").alias("Pickup_Zone"),
    col("DOLocationID").alias("Dropoff_Zone"),
    col("trip_count"),
    spark_round("avg_distance", 2).alias("avg_distance_miles"),
    spark_round("avg_duration_minutes", 2).alias("avg_duration_min"),
    spark_round("avg_fare", 2).alias("avg_fare_$")
).show(20)
 # Step 4: Trip Duration Analysis by Zone Pairs
print("\n4. TRIP DURATION ANALYSIS BY ZONE PAIRS")
print("-" * 45)

# Find routes with longest average duration
longest_duration_routes = routes_analysis \
    .filter(col("trip_count") >= 50) \
    .orderBy(desc("avg_duration_minutes"))

print("Routes with Longest Average Duration (min 50 trips):")
longest_duration_routes.select(
    col("PULocationID").alias("Pickup_Zone"),
    col("DOLocationID").alias("Dropoff_Zone"),
    col("trip_count"),
    spark_round("avg_duration_minutes", 2).alias("avg_duration_min"),
    spark_round("avg_distance", 2).alias("avg_distance_miles")
).show(15)
# Find routes with shortest average duration
shortest_duration_routes = routes_analysis \
    .filter(col("trip_count") >= 50) \
    .orderBy(asc("avg_duration_minutes"))

print("\nRoutes with Shortest Average Duration (min 50 trips):")
shortest_duration_routes.select(
    col("PULocationID").alias("Pickup_Zone"),
    col("DOLocationID").alias("Dropoff_Zone"),
    col("trip_count"),
    spark_round("avg_duration_minutes", 2).alias("avg_duration_min"),
    spark_round("avg_distance", 2).alias("avg_distance_miles")
).show(15)
# Step 5: Zone Activity Analysis
print("\n5. ZONE ACTIVITY ANALYSIS")
print("-" * 30)

# Combine pickup and dropoff activity for each zone
pickup_activity = pickup_zones.select(col("PULocationID").alias("LocationID"), col("pickup_count"))
dropoff_activity = dropoff_zones.select(col("DOLocationID").alias("LocationID"), col("dropoff_count"))

zone_activity = pickup_activity.join(dropoff_activity, "LocationID", "outer") \
    .fillna(0) \
    .select(
        col("LocationID"),
        col("pickup_count"),
        col("dropoff_count"),
        (col("pickup_count") + col("dropoff_count")).alias("total_activity")
    ) \
    .orderBy(desc("total_activity"))

print("Top 15 Most Active Zones (Pickup + Dropoff combined):")
zone_activity.show(15)
# Step 6: Distance vs Duration Analysis by Routes
print("\n6. DISTANCE VS DURATION EFFICIENCY ANALYSIS")
print("-" * 50)

# Calculate speed (distance/duration) for different routes
route_efficiency = routes_analysis \
    .filter((col("avg_distance") > 0) & (col("avg_duration_minutes") > 0)) \
    .select(
        col("PULocationID").alias("Pickup_Zone"),
        col("DOLocationID").alias("Dropoff_Zone"),
        col("trip_count"),
        spark_round("avg_distance", 2).alias("avg_distance_miles"),
        spark_round("avg_duration_minutes", 2).alias("avg_duration_min"),
        spark_round((col("avg_distance") / (col("avg_duration_minutes") / 60)), 2).alias("avg_speed_mph")
    ) \
    .filter(col("trip_count") >= 30)

print("Routes with Highest Average Speed (min 30 trips):")
route_efficiency.orderBy(desc("avg_speed_mph")).show(15)

print("\nRoutes with Lowest Average Speed (min 30 trips):")
route_efficiency.orderBy(asc("avg_speed_mph")).show(15)
# Step 7: Self-Zone Trips Analysis
print("\n7. SELF-ZONE TRIPS ANALYSIS")
print("-" * 35)

# Analyze trips within the same zone
same_zone_trips = trips_with_duration.filter(col("PULocationID") == col("DOLocationID")) \
    .groupBy("PULocationID") \
    .agg(
        count("*").alias("same_zone_trips"),
        avg("trip_distance").alias("avg_distance"),
        avg("trip_duration_minutes").alias("avg_duration"),
        avg("fare_amount").alias("avg_fare")
    ) \
    .orderBy(desc("same_zone_trips"))

print("Zones with Most Same-Zone Trips:")
same_zone_trips.select(
    col("PULocationID").alias("Zone_ID"),
    col("same_zone_trips"),
    spark_round("avg_distance", 2).alias("avg_distance_miles"),
    spark_round("avg_duration", 2).alias("avg_duration_min"),
    spark_round("avg_fare", 2).alias("avg_fare_$")
).show(15)
# Step 8: Prepare Aggregated Route Data for Visualization
print("\n8. PREPARING AGGREGATED ROUTE DATA FOR VISUALIZATION")
print("-" * 55)

# Create summary statistics for visualization team (Member 5)
route_summary = routes_analysis.select(
    col("PULocationID"),
    col("DOLocationID"),
    col("trip_count"),
    spark_round("avg_distance", 3).alias("avg_distance"),
    spark_round("avg_duration_minutes", 3).alias("avg_duration"),
    spark_round("avg_fare", 3).alias("avg_fare")
).filter(col("trip_count") >= 20)  # Filter for significant routes

print(f"Route summary prepared with {route_summary.count()} significant routes")
print("Sample of route summary data:")
route_summary.show(10)
# Member 3 - Geospatial and Route Analysis Code
