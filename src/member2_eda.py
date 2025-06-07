import os
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import avg, count, hour, dayofweek, dayofmonth, col
import matplotlib.pyplot as plt

# -------------------------------
# Create SparkSession
# -------------------------------
spark = SparkSession.builder \
    .appName("FareTipModeling") \
    .master("local[*]") \
    .getOrCreate()

# -------------------------------
# Read Parquet Files
# -------------------------------
folder_path = Path(r"D:\L4S2\Big_Data\Assignment\NYC_Taxi_Trip_Data_Analysis\data\clean\cleanedData1")
parquet_files = sorted(folder_path.glob("*.parquet"))
all_files_path = [str(f) for f in parquet_files]

df = spark.read.parquet(*all_files_path)
df.printSchema()

# -------------------------------
# Trip Statistics
# -------------------------------
avg_stats = df.select(
    avg("fare_amount").alias("average_fare"),
    avg("trip_distance").alias("average_distance")
)
avg_stats.show()

# -------------------------------
# Temporal Patterns
# -------------------------------
output_path = Path(r"D:\L4S2\Big_Data\Assignment\NYC_Taxi_Trip_Data_Analysis\output")
output_path.mkdir(parents=True, exist_ok=True)

df = df.withColumn("hour", hour("tpep_pickup_datetime")) \
       .withColumn("day_of_week", dayofweek("tpep_pickup_datetime")) \
       .withColumn("day_of_month", dayofmonth("tpep_pickup_datetime"))

# Hourly Trends
hourly_df = df.groupBy("hour").agg(
    avg("fare_amount").alias("avg_fare"),
    avg("trip_distance").alias("avg_distance"),
    count("*").alias("trip_count")
).orderBy("hour")

hourly_pd = hourly_df.toPandas()

plt.figure(figsize=(10, 5))
plt.plot(hourly_pd["hour"], hourly_pd["avg_fare"], marker='o')
plt.title("Average Fare by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Fare")
plt.grid(True)
plt.savefig(output_path / "avg_fare_by_hour.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(hourly_pd["hour"], hourly_pd["avg_distance"], marker='o', color='green')
plt.title("Average Trip Distance by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Distance")
plt.grid(True)
plt.savefig(output_path / "avg_distance_by_hour.png")
plt.close()

plt.figure(figsize=(10, 5))
plt.bar(hourly_pd["hour"], hourly_pd["trip_count"], color='orange')
plt.title("Trip Count by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Trips")
plt.grid(True)
plt.savefig(output_path / "trip_count_by_hour.png")
plt.close()

# Day of Week Trends
dow_df = df.groupBy("day_of_week").agg(
    avg("fare_amount").alias("avg_fare"),
    avg("trip_distance").alias("avg_distance"),
    count("*").alias("trip_count")
).orderBy("day_of_week")

dow_pd = dow_df.toPandas()
plt.figure(figsize=(10, 5))
plt.bar(dow_pd["day_of_week"], dow_pd["trip_count"], color='purple')
plt.title("Trip Count by Day of Week (1=Sun ... 7=Sat)")
plt.xlabel("Day of Week")
plt.ylabel("Number of Trips")
plt.grid(True)
plt.savefig(output_path / "trip_count_by_dayofweek.png")
plt.close()

# Day of Month Trends
dom_df = df.groupBy("day_of_month").agg(
    avg("fare_amount").alias("avg_fare"),
    avg("trip_distance").alias("avg_distance"),
    count("*").alias("trip_count")
).orderBy("day_of_month")

dom_pd = dom_df.toPandas()
plt.figure(figsize=(10, 5))
plt.bar(dom_pd["day_of_month"], dom_pd["trip_count"], color='teal')
plt.title("Trip Count by Day of Month")
plt.xlabel("Day of Month")
plt.ylabel("Number of Trips")
plt.grid(True)
plt.savefig(output_path / "trip_count_by_dayofmonth.png")
plt.close()

# -------------------------------
# Patterns by Passenger, Vendor, Distance
# -------------------------------
# Avg Fare by Passenger Count
passenger_df = df.groupBy("passenger_count").agg(
    avg("fare_amount").alias("avg_fare"),
    count("*").alias("trip_count")
).orderBy("passenger_count")

passenger_pd = passenger_df.toPandas()
plt.figure(figsize=(10, 5))
plt.bar(passenger_pd["passenger_count"], passenger_pd["avg_fare"], color="orange")
plt.title("Average Fare by Passenger Count")
plt.xlabel("Passenger Count")
plt.ylabel("Average Fare ($)")
plt.grid(axis='y')
plt.savefig(output_path / "avg_fare_by_passenger_count.png")
plt.close()

# Avg Fare by Vendor
vendor_df = df.groupBy("VendorID").agg(
    avg("fare_amount").alias("avg_fare"),
    count("*").alias("trip_count")
).orderBy("VendorID")

vendor_pd = vendor_df.toPandas()
plt.figure(figsize=(8, 5))
plt.bar(vendor_pd["VendorID"], vendor_pd["avg_fare"], color="skyblue")
plt.title("Average Fare by Vendor ID")
plt.xlabel("Vendor ID")
plt.ylabel("Average Fare ($)")
plt.grid(axis='y')
plt.savefig(output_path / "avg_fare_by_vendor.png")
plt.close()

# Fare vs Distance Scatter
scatter_sample_df = df.select("trip_distance", "fare_amount").dropna().filter(
    (col("trip_distance") > 0) & (col("trip_distance") < 50) &
    (col("fare_amount") > 0) & (col("fare_amount") < 300)
).sample(False, 0.001)

scatter_pd = scatter_sample_df.toPandas()
plt.figure(figsize=(10, 6))
plt.scatter(scatter_pd["trip_distance"], scatter_pd["fare_amount"], alpha=0.3, s=10)
plt.title("Fare Amount vs Trip Distance")
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Fare Amount ($)")
plt.grid(True)
plt.savefig(output_path / "fare_vs_distance_scatter.png")
plt.close()

# -------------------------------
# Busiest Pickup and Dropoff Zones
# -------------------------------
pickup_df = df.groupBy("PULocationID").agg(
    count("*").alias("pickup_count")
).orderBy(col("pickup_count").desc()).limit(10)

pickup_pd = pickup_df.toPandas()
plt.figure(figsize=(10, 5))
plt.bar(pickup_pd["PULocationID"].astype(str), pickup_pd["pickup_count"], color="green")
plt.title("Top 10 Busiest Pickup Zones")
plt.xlabel("Pickup Location ID")
plt.ylabel("Number of Trips")
plt.grid(axis='y')
plt.savefig(output_path / "top_10_pickup_zones.png")
plt.close()

dropoff_df = df.groupBy("DOLocationID").agg(
    count("*").alias("dropoff_count")
).orderBy(col("dropoff_count").desc()).limit(10)

dropoff_pd = dropoff_df.toPandas()
plt.figure(figsize=(10, 5))
plt.bar(dropoff_pd["DOLocationID"].astype(str), dropoff_pd["dropoff_count"], color="purple")
plt.title("Top 10 Busiest Dropoff Zones")
plt.xlabel("Dropoff Location ID")
plt.ylabel("Number of Trips")
plt.grid(axis='y')
plt.savefig(output_path / "top_10_dropoff_zones.png")
plt.close()

# -------------------------------
# End Spark session
# -------------------------------
spark.stop()
