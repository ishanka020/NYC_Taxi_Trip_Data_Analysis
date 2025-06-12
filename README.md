# NYC_Taxi_Trip_Data_Analysis

The vast amount of taxi trip data in New York City contains millions of entries that can’t be handled by traditional tools. But there is a lot of knowledge hidden within this mountain of information.

There are two challenges: 

    1. Data Overload – The huge size and disorder of the dataset make it difficult to analyze without a distributed system.
    2. Unexplored Questions – Questions like “When are tips most generous?”, “What’s the busiest pickup zone?”, or “Which routes are the most common?” are not easily answered without the right tools and techniques.

So, the problem we’re trying to solve is:

    How can we use distributed data processing to extract meaningful, real-world data from massive NYC taxi trip data?


## Project Objective
  
    The objective of this project is to extract meaningful information from the New York City Taxi Trip dataset using Apache Spark, a powerful framework for processing big data. Our team focuses on analyzing large-  scale trip data to understand taxi usage patterns, fare behavior, trip routes, and more.

### Aims
    
    To clean and process large real-world datasets efficiently using Spark
    To explore time-based and location-based taxi trends across NYC
    To build a simple predictive model to estimate tips
    To visualize findings to support urban mobility and transportation conclusions
This project demonstrates how big data tools, such as Spark, can help make sense of urban transportation behavior, supporting city planners, taxi operators, and researchers in informed decision-making.


## Data source and format 
original data source: https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

combined raw data : https://drive.google.com/drive/folders/1L3UNsBg6fvLHyhXDMnarJBxpOTCH-Es3?usp=drive_link

cleaned Data : https://drive.google.com/drive/folders/117FJB0kmu9Kp64pIj1xUpRt_NVcViFuD?usp=drive_link

## Key steps in the code 

Clean Data
    1. Combine NY Taxi trip Data of 12 months in 2024
        Read individual .parquet files for each month.
        Merge them into a single DataFrame.
        Optionally save the combined dataset for reuse.
    2. Clean The Data Set
        Remove Null or Missing Values
        Filter Out Invalid Values
        Check for unexpected characters or data types
        Save the cleaned dataset for future processing

Analyze Data Set For Passenger Count
    Derive Useful Information by grouping data by passenger_count
    Visualize information

Exploratory Data Analysis (EDA)

This part focuses on understanding the structure and behavior of NYC taxi trip data through data aggregation, pattern analysis, and visualization.
Key Steps in the Code:
    1. Initialize SparkSession
        ◦ Reuse the common Spark configuration (common.init_spark()), ensuring consistent environment setup across the project.
    2. Load Cleaned Data
        ◦ Read .parquet files from data/clean/cleanedData1 using Spark.
    3. Trip Statistics
        ◦ Compute basic statistics such as:
            ▪ Average fare (fare_amount)
            ▪ Average trip distance (trip_distance)
    4. Extract Temporal Features
        ◦ Derive time-based columns from pickup datetime:
            ▪ Hour of the day
            ▪ Day of the week
            ▪ Day of the month
    5. Analyze Temporal Patterns
        ◦ Group data to analyze:
            ▪ Average fare, distance, and trip counts by hour
            ▪ Same metrics by day of week and day of month
        ◦ Generate and save visualizations (.png) for each trend.
    6. Passenger Count and Vendor Analysis
        ◦ Analyze average fare and trip counts grouped by:
            ▪ Number of passengers
            ▪ Vendor ID
    7. Fare vs. Distance Analysis
        ◦ Create a scatter plot to explore the correlation between trip distance and fare.
    8. Save All Visualizations
        ◦ Output charts to output/member2_eda_output/ for reporting and presentation.


Predicting Tip Amount with models

1. Loaded and combined all cleaned Parquet files into a single DataFrame
2. Filtered out rows with invalid fare or tip amounts
3. Dropped total_amount column to avoid data leakage
4. Performed time-based feature extraction (hour, day of week, trip duration)
5. Encoded categorical column store_and_fwd_flag into numeric
6. Applied log and square root transformations to skewed features
7. Detected outliers using IQR method for key features
8. Engineered new features:
    Ratio features (e.g., tip per mile)
    Interaction features (e.g., fare × distance)
    Location-based flags (e.g., is_airport_trip)
    Temporal flags (e.g., is_weekend, is_night)
9. Assembled selected features into a vector using VectorAssembler
10. Scaled features using StandardScaler
11. Split data into training and testing sets
12. Trained and evaluated multiple regression models:
    Linear Regression
    Random Forest
    Gradient Boosted Trees
    Tuned GBT using TrainValidationSplit
13. Created a blended model using averaged predictions from RF and GBT
14. Evaluated all models using RMSE and R² metrics


Analysis and Visualizations 

1. Trips per Hour (Bar & Line Plot)
Extracted hour from pickup timestamps.

Counted number of trips per hour.

Visualized the data using both bar and line plots to understand hourly demand.

2. Average Tip by Trip Distance
Binned trips into distance ranges (e.g., 0–1 mi, 1–3 mi, etc.).

Calculated average tip amount per distance bin.

Visualized using a bar plot to observe tipping trends by distance.

3. Payment Method Distribution
Grouped trips by payment_type.

Counted the number of trips per method (e.g., card, cash).

Displayed with a bar plot for a clear overview of payment preferences.

4. Average Tip Percentage by Passenger Count
Calculated tip percentage: (tip / fare) * 100.

Grouped by passenger_count and calculated the average tip percentage.

Visualized to analyze tipping behavior based on passenger numbers.

5. Top 10 Pickup → Dropoff Pairs (Heatmap)
Grouped trips by pickup and dropoff IDs.

Selected the 10 most frequent pairs.

Created a heatmap to visualize these popular routes.

6. Revenue Heatmap by Hour and Day of Week
Extracted day of the week and hour from timestamps.

Summed up total_amount as revenue for each time slot.

Visualized using a heatmap to show peak revenue periods.

7. Most Efficient Fare Pickup Locations
Computed fare_per_mile for each trip.

Averaged this metric per pickup location.

Visualized top 10 locations offering the highest fare efficiency.

8. Average Passenger Count by Hour
Grouped trips by hour and calculated the average passenger_count.

Used a line plot to visualize rider trends throughout the day.

9. Top Routes Network Graph
Identified top 20 most frequent pickup-dropoff pairs.

Visualized them using a directed network graph (NetworkX) to show route flow and strength (trip count as edge weight).

 Output
All plots are saved in the output/ directory.


## Instructions to run the code 

Go to src -> member1_data_cleaning.py
Include your local folder path and replace it with Your-path
Ex: /Your-path/data/rawData


## Summary of results and conclusions 

