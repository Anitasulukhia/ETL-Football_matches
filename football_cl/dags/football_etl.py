from airflow import DAG
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.decorators import task
from airflow.utils.dates import days_ago
import requests

POSTGRES_CONN_ID = 'postgres_default'
API_URL = "https://api.football-data.org/v4/competitions/CL/matches"
API_KEY = "get your key" 

default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1)
}

with DAG(
    dag_id='football_etl_pipeline_requests',
    default_args=default_args,
    schedule_interval='@daily',
    catchup=False
) as dag:

    @task()
    def extract_football_data():
        headers = {"X-Auth-Token": API_KEY}

        response = requests.get(API_URL, headers=headers)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Failed to fetch football data: {response.status_code} - {response.text}")

    @task()
    def transform_football_data(football_data):
        """Transform the extracted football matches data."""
        transformed_data = []

        for match in football_data.get('matches', []):  
            transformed_match = {
                'match_id': match['id'],
                'home_team': match['homeTeam']['name'],
                'away_team': match['awayTeam']['name'],
                'home_score': match['score']['fullTime'].get('home'),
                'away_score': match['score']['fullTime'].get('away'),
                'status': match['status'],
                'match_day': match['matchday'],
                'season': football_data.get('season', {}).get('id'),  
                'utc_date': match['utcDate']
            }
            transformed_data.append(transformed_match)

        return transformed_data

    @task()
    def load_football_data(transformed_data):
        """Load transformed football data into PostgreSQL."""
        pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
        conn = pg_hook.get_conn()
        cursor = conn.cursor()

        # Create table if it doesn't exist
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS football_matches (
                match_id INTEGER PRIMARY KEY,
                home_team VARCHAR(100),
                away_team VARCHAR(100),
                home_score INTEGER,
                away_score INTEGER,
                status VARCHAR(20),
                match_day INTEGER,
                season INTEGER,
                utc_date TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Insert transformed data into the table
        for match in transformed_data:
            cursor.execute("""
                INSERT INTO football_matches 
                (match_id, home_team, away_team, home_score, away_score, 
                status, match_day, season, utc_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (match_id) DO UPDATE 
                SET home_score = EXCLUDED.home_score,
                    away_score = EXCLUDED.away_score,
                    status = EXCLUDED.status;
            """, (
                match['match_id'],
                match['home_team'],
                match['away_team'],
                match['home_score'],
                match['away_score'],
                match['status'],
                match['match_day'],
                match['season'],
                match['utc_date']
            ))

        conn.commit()
        cursor.close()

    ## DAG Workflow - ETL Pipeline
    football_data = extract_football_data()
    transformed_data = transform_football_data(football_data)
    load_football_data(transformed_data)
