
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import hour, count, col, when, avg, date_format, sum
 
PARQUET_PATH = "E:\UOM_FIT\L4S2\Big_Data\NYC_Taxi_Trip_Data_Analysis\data"

spark = SparkSession.builder.appName("NYC Taxi Analysis").getOrCreate()

df = spark.read.parquet(PARQUET_PATH)
df.printSchema()
df.show(5)

# 1. Trips per Hour (Bar Plot)
trips_per_hour = df.withColumn("hour", hour("tpep_pickup_datetime")) \
                   .groupBy("hour") \
                   .agg(count("*").alias("trip_count")) \
                   .orderBy("hour")

trips_per_hour_pd = trips_per_hour.toPandas()
plt.figure(figsize=(12, 6))
sns.barplot(x="hour", y="trip_count", data=trips_per_hour_pd, palette="Blues_d")
plt.title("Number of Taxi Trips per Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Trips")
plt.savefig("plots/trips_per_hour_bar.png")
plt.close()

# 2. Trips per Hour (Line Plot)
df_hour = df.withColumn("hour", hour("tpep_pickup_datetime")) \
            .groupBy("hour").count().orderBy("hour")
df_hour_pd = df_hour.toPandas()

plt.figure(figsize=(10, 6))
sns.lineplot(data=df_hour_pd, x="hour", y="count", marker='o')
plt.title("Number of Taxi Trips by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Trips")
plt.grid(True)
plt.savefig("plots/trips_per_hour_line.png")
plt.close()

# 3. Average Tip by Trip Distance
df = df.withColumn("distance_bin", when(col("trip_distance") <= 1, "0-1 mi")
                   .when(col("trip_distance") <= 3, "1-3 mi")
                   .when(col("trip_distance") <= 5, "3-5 mi")
                   .when(col("trip_distance") <= 10, "5-10 mi")
                   .otherwise("10+ mi"))

tip_df = df.groupBy("distance_bin").agg(avg("tip_amount").alias("avg_tip"))
tip_pd = tip_df.toPandas().sort_values("distance_bin")

plt.figure(figsize=(8, 6))
sns.barplot(x="distance_bin", y="avg_tip", data=tip_pd)
plt.title("Average Tip by Trip Distance")
plt.xlabel("Trip Distance Bin")
plt.ylabel("Average Tip Amount")
plt.savefig("plots/avg_tip_by_distance.png")
plt.close()

# 4. Payment Method Distribution
df_payment = df.groupBy("payment_type").count().orderBy("count", ascending=False)
df_payment_pd = df_payment.toPandas()

plt.figure(figsize=(8, 6))
sns.barplot(data=df_payment_pd, x="payment_type", y="count")
plt.title("Distribution of Payment Methods")
plt.xlabel("Payment Type")
plt.ylabel("Trip Count")
plt.savefig("plots/payment_distribution.png")
plt.close()

# 5. Average Tip % by Passenger Count
df_tip = df.withColumn("tip_percentage",
                       when(col("fare_amount") > 0, col("tip_amount") / col("fare_amount") * 100).otherwise(0)) \
           .groupBy("passenger_count") \
           .agg(avg("tip_percentage").alias("avg_tip_percentage")) \
           .orderBy("passenger_count")

df_tip_pd = df_tip.toPandas()

plt.figure(figsize=(10, 6))
sns.barplot(data=df_tip_pd, x="passenger_count", y="avg_tip_percentage")
plt.title("Average Tip Percentage by Passenger Count")
plt.xlabel("Passenger Count")
plt.ylabel("Average Tip %")
plt.savefig("plots/tip_percentage_by_passenger.png")
plt.close()

# 6. Top 10 Pickup , Dropoff Pairs (Heatmap)
df_locations = df.groupBy("PULocationID", "DOLocationID").count().orderBy("count", ascending=False).limit(10)
df_locations_pd = df_locations.toPandas()

pivot_table = df_locations_pd.pivot(index="PULocationID", columns="DOLocationID", values="count")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, annot=True, fmt="g")
plt.title("Top 10 Most Frequent Pickup → Dropoff Pairs")
plt.xlabel("Dropoff Location ID")
plt.ylabel("Pickup Location ID")
plt.savefig("plots/top_routes_heatmap.png")
plt.close()

# 7. Revenue Heatmap by Hour and Day
df = df.withColumn("hour", hour("tpep_pickup_datetime"))
df = df.withColumn("day_of_week", date_format("tpep_pickup_datetime", "E"))

revenue_df = df.groupBy("day_of_week", "hour").agg(sum("total_amount").alias("total_revenue"))
revenue_pd = revenue_df.toPandas().pivot(index="day_of_week", columns="hour", values="total_revenue")

plt.figure(figsize=(12, 6))
sns.heatmap(revenue_pd, cmap="YlGnBu")
plt.title("Trip Revenue Heatmap by Hour and Day of Week")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week")
plt.savefig("plots/revenue_heatmap.png")
plt.close()

# 8. Most Efficient Fare Pickup Locations
df = df.withColumn("fare_per_mile", col("fare_amount") / (col("trip_distance") + 0.01))
eff_df = df.groupBy("PULocationID").agg(avg("fare_per_mile").alias("avg_fare_per_mile"))
eff_pd = eff_df.toPandas().sort_values("avg_fare_per_mile", ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(x="avg_fare_per_mile", y="PULocationID", data=eff_pd)
plt.title("Top 10 Most Efficient Fare Pickup Locations")
plt.xlabel("Average Fare per Mile")
plt.ylabel("Pickup Location ID")
plt.savefig("plots/efficient_pickup_locations.png")
plt.close()

# 9. Avg Passenger Count by Hour
passenger_df = df.groupBy("hour").agg(avg("passenger_count").alias("avg_passenger_count"))
passenger_pd = passenger_df.toPandas().sort_values("hour")

plt.figure(figsize=(10, 6))
sns.lineplot(x="hour", y="avg_passenger_count", data=passenger_pd, marker='o')
plt.title("Average Passenger Count by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Passenger Count")
plt.grid(True)
plt.savefig("plots/avg_passenger_by_hour.png")
plt.close()

# 10. Top Routes Network Graph
import networkx as nx

top_routes = df.groupBy("PULocationID", "DOLocationID").agg(count("*").alias("trip_count")) \
               .orderBy(col("trip_count").desc()).limit(20).toPandas()

G = nx.DiGraph()
for _, row in top_routes.iterrows():
    G.add_edge(row["PULocationID"], row["DOLocationID"], weight=row["trip_count"])

plt.figure(figsize=(14, 10))
pos = nx.kamada_kawai_layout(G)
weights = [G[u][v]['weight'] / 100 for u, v in G.edges()]
nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=700)
nx.draw_networkx_edges(G, pos, edge_color='gray', width=weights, arrows=True, arrowstyle='-|>', arrowsize=15)
nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
plt.title("Top 20 Most Frequent Pickup - Dropoff Routes", fontsize=14)
plt.axis('off')
plt.tight_layout()
plt.savefig("plots/top_routes_graph.png")
plt.close()

print("Analysis Complete. All plots saved in 'plots/' directory.")
