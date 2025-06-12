from pathlib import Path
import matplotlib.pyplot as plt
from pyspark.sql.functions import col, hour, dayofweek, dayofmonth, avg, count

from common import init_spark, get_project_paths

# -------------------------------------
# Init Spark and Paths
# -------------------------------------
spark = init_spark("Member2 EDA")
print("Spark version:", spark.version)

# data_path, output_path = get_project_paths()
# output_path = output_path / "member2_eda_output"
# output_path.mkdir(parents=True, exist_ok=True)

# parquet_files = sorted(data_path.glob("*.parquet"))
df = spark.read.parquet("/Your-path/data/cleanedData")

# -------------------------------------
# 1. Trip Statistics: Average fare, distance
# -------------------------------------

df.printSchema()

avg_stats = df.select(
    avg("fare_amount").alias("average_fare"),
    avg("trip_distance").alias("average_distance")
)

avg_stats.show()

# -------------------------------------
# Add Time Features
# -------------------------------------
df = df.withColumn("hour", hour("tpep_pickup_datetime")) \
       .withColumn("day_of_week", dayofweek("tpep_pickup_datetime")) \
       .withColumn("day_of_month", dayofmonth("tpep_pickup_datetime"))

# -------------------------------------
# Hourly Trends
# -------------------------------------
hourly_df = df.groupBy("hour").agg(
    avg("fare_amount").alias("avg_fare"),
    avg("trip_distance").alias("avg_distance"),
    count("*").alias("trip_count")
).orderBy("hour")
hourly_pd = hourly_df.toPandas()

# -------------------------------------
# 2. Temporal Patterns: Hourly, Daily, Weekly Trends
# -------------------------------------

# Plot: Average fare by hour
plt.figure(figsize=(10, 5))
plt.plot(hourly_pd["hour"], hourly_pd["avg_fare"], marker='o')
plt.title("Average Fare by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Fare")
plt.grid(True)
plt.savefig(output_path / "avg_fare_by_hour.png")
plt.close()

# Plot: Average distance by hour
plt.figure(figsize=(10, 5))
plt.plot(hourly_pd["hour"], hourly_pd["avg_distance"], marker='o', color='green')
plt.title("Average Trip Distance by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Distance")
plt.grid(True)
plt.savefig(output_path / "avg_distance_by_hour.png")
plt.close()

# Plot: Trip count by hour
plt.figure(figsize=(10, 5))
plt.bar(hourly_pd["hour"], hourly_pd["trip_count"], color='orange')
plt.title("Trip Count by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Trips")
plt.grid(True)
plt.savefig(output_path / "trip_count_by_hour.png")
plt.close()

# -------------------------------------
# Daily of Week Trends
# -------------------------------------
dow_df = df.groupBy("day_of_week").agg(
    avg("fare_amount").alias("avg_fare"),
    avg("trip_distance").alias("avg_distance"),
    count("*").alias("trip_count")
).orderBy("day_of_week")
dow_pd = dow_df.toPandas()

# Plot: Trip count by day of week
plt.figure(figsize=(10, 5))
plt.bar(dow_pd["day_of_week"], dow_pd["trip_count"], color='purple')
plt.title("Trip Count by Day of Week (1=Sun ... 7=Sat)")
plt.xlabel("Day of Week")
plt.ylabel("Trips")
plt.grid(True)
plt.savefig(output_path / "trip_count_by_dayofweek.png")
plt.close()

# -------------------------------------
# Day of Month Trends
# -------------------------------------
dom_df = df.groupBy("day_of_month").agg(
    avg("fare_amount").alias("avg_fare"),
    avg("trip_distance").alias("avg_distance"),
    count("*").alias("trip_count")
).orderBy("day_of_month")
dom_pd = dom_df.toPandas()

# Plot: Trip count by day of month
plt.figure(figsize=(10, 5))
plt.bar(dom_pd["day_of_month"], dom_pd["trip_count"], color='teal')
plt.title("Trip Count by Day of Month")
plt.xlabel("Day")
plt.ylabel("Trips")
plt.grid(True)
plt.savefig(output_path / "trip_count_by_dayofmonth.png")
plt.close()

# -------------------------------------
# 3. Patterns by Passenger Count, Vendor, Distance
# -------------------------------------

# -------------------------------------
# Average Fare by Passenger Count
# -------------------------------------
passenger_df = df.groupBy("passenger_count").agg(
    avg("fare_amount").alias("avg_fare"),
    count("*").alias("trip_count")
).orderBy("passenger_count")
passenger_pd = passenger_df.toPandas()

plt.figure(figsize=(10, 5))
plt.bar(passenger_pd["passenger_count"], passenger_pd["avg_fare"], color="orange")
plt.title("Average Fare by Passenger Count")
plt.xlabel("Passengers")
plt.ylabel("Avg Fare ($)")
plt.grid(True)
plt.savefig(output_path / "avg_fare_by_passenger_count.png")
plt.close()

# -------------------------------------
# Average Fare by Vendor
# -------------------------------------
vendor_df = df.groupBy("VendorID").agg(
    avg("fare_amount").alias("avg_fare"),
    count("*").alias("trip_count")
).orderBy("VendorID")
vendor_pd = vendor_df.toPandas()

plt.figure(figsize=(8, 5))
plt.bar(vendor_pd["VendorID"], vendor_pd["avg_fare"], color="skyblue")
plt.title("Average Fare by Vendor ID")
plt.xlabel("Vendor ID")
plt.ylabel("Avg Fare ($)")
plt.grid(True)
plt.savefig(output_path / "avg_fare_by_vendor.png")
plt.close()

# -------------------------------------
# Fare vs. Distance (Scatter Plot)
# -------------------------------------
scatter_df = df.select("trip_distance", "fare_amount").dropna().filter(
    (col("trip_distance") > 0) & (col("trip_distance") < 50) &
    (col("fare_amount") > 0) & (col("fare_amount") < 300)
).sample(False, 0.001)

scatter_pd = scatter_df.toPandas()

plt.figure(figsize=(10, 6))
plt.scatter(scatter_pd["trip_distance"], scatter_pd["fare_amount"], alpha=0.3, s=10)
plt.title("Fare vs. Trip Distance")
plt.xlabel("Distance (miles)")
plt.ylabel("Fare ($)")
plt.grid(True)
plt.savefig(output_path / "fare_vs_distance_scatter.png")
plt.close()

# -------------------------------------
# Stop Spark
# -------------------------------------

spark.stop()
