# Football Data ETL and Score Prediction
Developed ETL processes and data pipelines to automate data extraction, transformation, and loading using Airflow. Analyzed historical match data, applying machine learning algorithms to predict match scores and gain insights into team performance.

# Project Overview

This project automates the extraction, transformation, and loading (ETL) of football match data using Apache Airflow and PostgreSQL, with machine learning models applied to predict match scores based on historical data. Data is extracted from a free API that provides both historical and future football match data. The entire setup is containerized using Docker for easy deployment and scalability. Additionally, ASTRO is used for managing Apache Airflow deployments.

# Features

ETL Pipeline:
Built with Apache Airflow, this pipeline automates the extraction of match data from an external free API, transforms it for analysis, and loads it into a PostgreSQL database.
Machine Learning:
Machine learning models are trained to predict match scores and analyze team performance based on historical data.
Data Storage:
All match data is stored in a PostgreSQL database, enabling efficient querying and analysis.
Docker Containers:
The application, including Airflow, PostgreSQL, and the app itself, is containerized using Docker to ensure a consistent and scalable environment.
ASTRO:
Used for managing Apache Airflow deployments, providing a simple interface to manage DAGs and schedules in the cloud or on-premises.
Automation:
The full pipeline is automated, reducing manual intervention while ensuring continuous model updates and match predictions.

# Technologies Used:

Python: For data processing, machine learning, and automation.
Apache Airflow: For orchestrating the ETL pipeline and automating data workflows.
ASTRO: For managing Apache Airflow deployments.
PostgreSQL: For storing and querying match data.
Docker: For containerizing the application and ensuring consistent environments.
Scikit-learn: For building and evaluating machine learning models.
Pandas: For data manipulation and analysis.
SQLAlchemy: For handling database interactions within the Python environment.
Football API: Free API for extracting both historical and future football match data.
Project Structure

# The project is divided into two main parts:

Data Pipeline (football_cl):
This folder contains the Python code for implementing the Airflow DAGs and the ETL pipeline. The core code for orchestrating the workflow is located in the dag folder.
Machine Learning (predict_football_scores):
This folder contains the Python implementation for the machine learning models and the logic to read data from the PostgreSQL database. It handles data preprocessing, model training, and score prediction.
Folder Structure
football_cl
dag: Contains the Airflow DAG definitions for ETL, including tasks for data extraction, transformation, and loading into PostgreSQL.
predict_football_scores
Contains Python scripts to load match data from the database, process it, train machine learning models, and predict future match scores.

# if you want to implement this project:

1. Clone the repository
git clone <repository_url>
cd <project_directory>

2. Build and start Docker containers
Make sure Docker is installed. Then, run the following command to build and start the containers:

docker-compose up --build
This will:

Set up the PostgreSQL container and initialize the database.
Set up the Apache Airflow container and start the scheduler and web server.

3. Install Python dependencies
If running outside of Docker, install the required dependencies:

pip install -r requirements.txt

4. Set up PostgreSQL

5. Configure Airflow (with ASTRO)
Access the Airflow web UI at http://localhost:8080.
In the football_cl/dag folder, you’ll find the Airflow DAGs that define the ETL process.
Use ASTRO to deploy and schedule the DAGs in the cloud or locally.

7. Run the ETL pipeline
The ETL pipeline can be triggered manually or scheduled using Airflow. This process will extract data from the free API, load it into PostgreSQL, and trigger model training.

8. Machine Learning Model Training and Prediction
Once the data is loaded into PostgreSQL, the machine learning code in the predict_football_scores folder reads the data and trains models to predict future match scores.
The models can be retrained periodically with new data to improve prediction accuracy.
Docker Configuration

The project uses docker-compose to manage the containers:

PostgreSQL: Database container for storing match data.
Airflow: Orchestrates the ETL pipeline.
App: Contains the Python application for data processing and machine learning.
The docker-compose.yml file defines the services, volumes, and networks needed for the project.

Example Workflow:
Data Extraction:
The ETL pipeline extracts football match data from a free API that provides both historical and future match data.
Data Transformation:
The data is cleaned, processed, and transformed into a suitable format for storage in PostgreSQL.
Data Loading:
Cleaned data is inserted into the PostgreSQL database.
Machine Learning:
Historical data from PostgreSQL is used to train machine learning models for predicting future match scores.
Automation:
The full ETL process is automated using Airflow, with regular model updates based on new data.
