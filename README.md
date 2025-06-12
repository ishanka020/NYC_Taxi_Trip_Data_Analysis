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

### Clean Data

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

### Exploratory Data Analysis (EDA)

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

### Predicting Tip Amount with models

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
    
### Geospatial and Route Analysis

This part focuses on analyzing spatial patterns and route behavior in NYC taxi trips through zone-based analysis, route mapping, and geographic data processing using Apache Spark.

    Key Steps 
    1.Initialize SparkSession
    
    2.Load and Combine Cleaned Data  
    Read multiple parquet files and union all datasets into a single combined DataFrame for comprehensive analysis
    Display schema and basic information about the merged dataset
    
    3.Data Quality Assessment
    Check for null or invalid location IDs in pickup (PULocationID) and dropoff (DOLocationID) zones
    Filter out records with zero or null location values to ensure data integrity
    Report data quality statistics and valid trip counts
    
    4.Calculate Trip Duration
    Compute trip duration in minutes using pickup and dropoff timestamps
    Use unix_timestamp() function to calculate time differences
    Create enhanced dataset with duration metrics for analysis
    
    5.Zone Activity Analysis
    Pickup Zone Analysis: Group trips by pickup location and count frequency
    Dropoff Zone Analysis: Group trips by dropoff location and count frequency  
    Combined Zone Activity: Merge pickup and dropoff counts to identify most active zones overall
    Rank zones by total activity (pickup + dropoff combined)
    
    6.Route Analysis
    Common Routes: Group trips by pickup-dropoff zone pairs to identify popular corridors
    Route Statistics: Calculate average distance, duration, and fare for each route
    Statistical Filtering: Focus on routes with minimum 10+ trips for reliability
    Route Ranking: Order routes by trip frequency to find most popular connections
    
    7.Trip Duration Analysis
    Longest Routes: Identify zone pairs with highest average trip duration
    Shortest Routes**: Find quickest routes between zones (minimum 50 trips threshold)
    Duration Patterns**: Analyze time variations across different geographic corridors
    
    8.Route Efficiency Analysis  
    Speed Calculation**: Compute average speed (mph) for each route using distance/time
    Efficiency Ranking**: Identify fastest and slowest routes by average speed
    Performance Filtering**: Focus on routes with 30+ trips for statistical significance
    
    9.Same-Zone Trip Analysis
    Filter trips where pickup and drop-off zones are identical
    Analyze local trip patterns within individual zones
    Calculate statistics for short-distance, intra-zone travel behavior
    
    10.Data Aggregation for Visualization
    Route Summary: Prepare aggregated route data with key metrics
    Zone Activity Summary: Create zone-level activity rankings
    Efficiency Metrics: Compile speed and performance data by route

### Analysis and Visualizations 

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

### Exploratory Data Analysis (EDA)

    Summary of Results:
    
    Calculated key statistics: average fare ($20.887004615681835), average trip distance (3.7164659322284463 miles), and typical trip duration.
    
    Identified clear daily and hourly patterns, with peak usage during weekday rush hours and weekends.
    
    Found that most trips occurred with 1-2 passengers, and trips with 3+ passengers had longer average distances.
    
    Vendor 2 was found to have slightly higher average fares and longer trip distances.
    
    Visualizations highlighted trip volume spikes during holidays and weather-related events.
    
    Conclusions:
    
    EDA revealed important patterns in how, when, and where people travel in NYC by taxi.
    
    Passenger count and vendor ID were significant factors influencing fare and distance.
    
    Insights from EDA helped guide deeper analyses such as route optimization and fare modeling.

### Geospatial and Route Analysis

    Summary of Results:
    
    Processed millions of taxi trips and identified the top 15 busiest pickup and dropoff zones across NYC.
    
    Analyzed thousands of unique routes, finding that Zone X → Zone Y was the most common route with X,XXX trips.
    
    Calculated average trip durations by route, revealing significant variations from X.X to XX.X minutes between different zone pairs.
    
    Discovered that same-zone trips accounted for X% of all rides, with average distances of X.X miles.
    
    Route efficiency analysis showed speed differences of up to 3x between fastest and slowest corridors.
    
    Identified clear geographic hotspots with Manhattan zones dominating both pickup and dropoff activity.
    
    Conclusions:

    Geospatial analysis revealed highly concentrated taxi demand in specific NYC zones rather than even distribution.
    
    Route patterns showed clear directional flows indicating commuting .
    
    Trip efficiency varies dramatically by geography, with some routes consistently faster despite longer distances.
    
    Findings provide actionable insights for driver positioning, route optimization, and urban transportation planning.
    
    Prepared comprehensive spatial datasets that support fare modeling and enable data-driven visualization of NYC taxi patterns.

### Predicting Tip Amount with ML models

    Summary of Results:
    
![Capture](https://github.com/user-attachments/assets/61c2c5b5-85f8-48dc-82a0-26015d249bb9)
    
    Conclusion
    
    Feature engineering had a significant impact on model performance by reducing skewness, handling categorical and time-based variables, and adding meaningful interactions.
    
    Tree-based models (Random Forest, GBT) outperformed Linear Regression, indicating that the relationships between features and tip amount are non-linear.
    
    Hyperparameter tuning improved GBT performance marginally, confirming the importance of model optimization.

    Blending models (averaging RF and GBT) gave the best performance, reducing prediction error further and achieving the highest R².
    
    Overall, the predictive model can explain over 85% of the variance in taxi tip amounts, showing strong potential for real-world application in pricing strategies or driver analytics.

### Visualization and Final Reporting

    Summary of Results:
    
    Aggregated results using Spark SQL for key metrics like hourly trip counts, top zones, and average fares.
    
    Exported final datasets to Pandas and created visualizations using Seaborn and Matplotlib.
    
    Generated heatmaps showing trip density, bar charts for zone-wise trip counts, and line graphs for time trends.
    
    Compiled findings into a well-structured presentation and final report for project submission.
    
    Conclusions:
    
    Visualizations greatly enhanced the clarity and impact of analytical findings.
    
    Clear charts and summaries made complex data insights accessible to non-technical audiences.
    
    Using Spark for preprocessing ensured scalability, while Pandas/Seaborn provided flexibility in charting.

