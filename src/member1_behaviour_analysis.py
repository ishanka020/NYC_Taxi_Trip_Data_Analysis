from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, unix_timestamp, avg, expr
import matplotlib.pyplot as plt
import seaborn as sns

# Start Spark Session
spark = SparkSession.builder.appName("PassengerBehaviorAnalysis").getOrCreate()

# Load cleaned data
df_new = spark.read.parquet("/home/ariso/Documents/bigDataAssignment/NYC_Taxi_Trip_Data_Analysis/data/cleanedData")


# 1. Calculate trip duration in minutes
df_new = df_new.withColumn(
    "trip_duration_min",
    (unix_timestamp(col("tpep_dropoff_datetime")) - unix_timestamp(col("tpep_pickup_datetime"))) / 60
)

# 2. Create ride_type column: 'Solo' if passenger_count == 1 else 'Group'
df_new = df_new.withColumn(
    "ride_type",
    when(col("passenger_count") == 1, "Solo").otherwise("Group")
)

# 3. Aggregate stats by ride_type (Solo vs Group)
agg_ride_type = df_new.groupBy("ride_type").agg(
    avg("fare_amount").alias("avg_fare"),
    avg("trip_distance").alias("avg_distance"),
    avg("trip_duration_min").alias("avg_duration_min"),
    avg("tip_amount").alias("avg_tip")
)

agg_ride_type.show()

# 4. Aggregate stats by passenger_count (detailed segmentation)
agg_passenger_count = df_new.groupBy("passenger_count").agg(
    avg("fare_amount").alias("avg_fare"),
    avg("trip_distance").alias("avg_distance"),
    avg("trip_duration_min").alias("avg_duration_min"),
    avg("tip_amount").alias("avg_tip")
).orderBy("passenger_count")

agg_passenger_count.show()

# 5. Calculate average tip ratio (tip_amount / fare_amount) by passenger_count for visualization
df_tip_ratio = df_new.withColumn("tip_ratio", expr("tip_amount / fare_amount")) \
    .groupBy("passenger_count") \
    .agg(avg("tip_ratio").alias("avg_tip_ratio")) \
    .orderBy("passenger_count")

# 6. Collect aggregated tip ratio to Pandas for plotting
pdf_tip_ratio = df_tip_ratio.toPandas()

# 7. Plot average tip ratio by passenger_count using Seaborn
plt.figure(figsize=(8, 5))
sns.barplot(x="passenger_count", y="avg_tip_ratio", data=pdf_tip_ratio)
plt.title("Average Tip Ratio by Passenger Count")
plt.xlabel("Passenger Count")
plt.ylabel("Average Tip / Fare Ratio")
plt.show()

# 8. Collect average distance data to Pandas for plotting
pdf_distance = agg_passenger_count.select("passenger_count", "avg_distance").toPandas()

# 9. Plot average distance by passenger count using Seaborn
plt.figure(figsize=(8, 5))
sns.barplot(x="passenger_count", y="avg_distance", data=pdf_distance)
plt.title("Average Trip Distance by Passenger Count")
plt.xlabel("Passenger Count")
plt.ylabel("Average Trip Distance (miles)")
plt.show()