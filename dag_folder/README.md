Overview
=============

This project explores the world of data engineering with ETL pipeline using astro airflow. We are going to use weather data from free api available from VisualCrossing and going to create a temperature prediction model based on past 7 days trend with other weather features such as pressure, humidity, temperature and windspeed.

Approach to the project
===========================

First of all lets create a dag. The dag will contain 4 parts:
1. Create a database table if not already exists
2. Check if 7 days data present
3. If 7 days data present then the pipeline will fetch today weather data
4. Else the pipeline will fetch past 8 days data
5. Desired features from the fetched data are extracted (date, pressure, wind_speed, temperature, pressure) and loaded to database.

Below image represent the DAG pipeline and show a successful run of fetching todays data.

![Screenshot from 2025-03-25 00-55-19](https://github.com/user-attachments/assets/59b4c1df-9615-4a49-b9f4-73d374cd6c45)

Each task clearly display what task they are performing.

Optimization achieved in this pipeline
=========================================

First iteration of this pipeline development lead to a pipeline where pipeline fetches data for current days data which increased the pipeline execution time. The above pipeline is the optimized pipeline where pipeline is forked and a dummy operator is added which reduced the exeuction time by ~50% whenever last 7 days present.

Deployment followed
=====================

Now lets follow the steps followed for the deployment step. I followed AWS free-tier services for deploying the servies.
1. Created a postgres rds service in AWS and enabled public access for it.
2. Create a MWAA (This is AWS managed Airflow) service. Here i created a S3 bucket where I uploaded the python dag file and then i created the MWAA service and selected the dag file from the S3 bucket. AWS will self manage this airflow workflow. HTTP connection (for fetching data using VisualCrossing api) and Postgres connection (for connecting with the Postgres rds service) are setup using the admin panel present in airflow ui.
3. This helped establish the connection between the Airflow dag and the Postgres database.
