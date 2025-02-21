import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import psycopg2
from psycopg2 import Error


def fetch_the_data():
    try:
        conn = psycopg2.connect(
            dbname="",
            user="",
            password="",
            host="",
            port=""
        )
        cur = conn.cursor()
        cur.execute("""
            SELECT match_id, home_team, away_team, 
                   CAST(home_score AS INTEGER),
                   CAST(away_score AS INTEGER),
                   status, match_day, season, 
                   utc_date, created_at
            FROM football_matches
        """)
        rows = cur.fetchall()
        columns = ['match_id', 'home_team', 'away_team', 'home_score',
                   'away_score', 'status', 'match_day', 'season',
                   'utc_date', 'created_at']

        with open("rawdata.csv", "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows(rows)

    except (Exception, Error) as error:
        print(f"Error while connecting to PostgreSQL: {error}")
        return None
    finally:
        if conn:
            cur.close()
            conn.close()
    return 'rawdata.csv'


def calculate_team_stats(df):
    team_stats = {}
    EPSILON = 0.0001

    df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
    df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')

    for team in set(df['home_team'].unique()) | set(df['away_team'].unique()):
        home_games = df[df['home_team'] == team].copy()
        away_games = df[df['away_team'] == team].copy()

        # Calculate average goals scored and conceded
        home_goals_scored = home_games['home_score'].mean() if len(home_games) > 0 else 0
        away_goals_scored = away_games['away_score'].mean() if len(away_games) > 0 else 0
        avg_goals_scored = (home_goals_scored + away_goals_scored) / 2 if len(home_games) > 0 or len(
            away_games) > 0 else 0

        home_goals_conceded = home_games['away_score'].mean() if len(home_games) > 0 else 0
        away_goals_conceded = away_games['home_score'].mean() if len(away_games) > 0 else 0
        avg_goals_conceded = (home_goals_conceded + away_goals_conceded) / 2 if len(home_games) > 0 or len(
            away_games) > 0 else EPSILON

        # Calculate win ratio
        home_wins = len(home_games[home_games['home_score'] > home_games['away_score']])
        away_wins = len(away_games[away_games['away_score'] > away_games['home_score']])
        total_games = len(home_games) + len(away_games)
        win_ratio = (home_wins + away_wins) / total_games if total_games > 0 else 0

        team_stats[team] = {
            'avg_goals_scored': float(avg_goals_scored),
            'avg_goals_conceded': float(max(avg_goals_conceded, EPSILON)),
            'win_ratio': float(win_ratio)
        }

    return team_stats


def clean_csv(input_filename, output_filename):
    try:
        df = pd.read_csv(input_filename,
                         dtype={
                             'match_id': str,
                             'home_team': str,
                             'away_team': str,
                             'status': str,
                             'season': str,
                             'utc_date': str,
                             'created_at': str
                         })

        df['home_score'] = pd.to_numeric(df['home_score'], errors='coerce')
        df['away_score'] = pd.to_numeric(df['away_score'], errors='coerce')
        df['match_day'] = pd.to_numeric(df['match_day'], errors='coerce')

        # Drop rows with missing scores
        df = df.dropna(subset=['home_score', 'away_score'])

        # Calculate team statistics using historical data
        team_stats = calculate_team_stats(df)

        # Add team statistics as features
        df['home_team_avg_goals'] = df['home_team'].map(lambda x: team_stats[x]['avg_goals_scored'])
        df['home_team_def_strength'] = df['home_team'].map(lambda x: 1 / team_stats[x]['avg_goals_conceded'])
        df['home_team_win_ratio'] = df['home_team'].map(lambda x: team_stats[x]['win_ratio'])

        df['away_team_avg_goals'] = df['away_team'].map(lambda x: team_stats[x]['avg_goals_scored'])
        df['away_team_def_strength'] = df['away_team'].map(lambda x: 1 / team_stats[x]['avg_goals_conceded'])
        df['away_team_win_ratio'] = df['away_team'].map(lambda x: team_stats[x]['win_ratio'])

        # Create target variable (1 for home win, 0 for away win or draw)
        df['match_outcome'] = (df['home_score'] > df['away_score']).astype(int)

        # Remove the actual scores as they wouldn't be available for prediction
        df = df.drop(columns=['home_score', 'away_score'])

        # Handle missing values with proper type inference
        for column in df.columns:
            if df[column].dtype == 'object':
                df[column] = df[column].fillna('')
            else:
                df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0)

        # Infer proper types after filling nulls
        df = df.infer_objects(copy=False)


        df.to_csv(output_filename, index=False)
        return df, team_stats
    except Exception as e:
        print(f"Error cleaning data: {e}")
        import traceback
        traceback.print_exc()
        return None, None
def train_model(df):
    try:
        # Drop non-feature columns
        df = df.drop(columns=["match_id", "created_at", "utc_date"], errors="ignore")

        # Encode categorical variables
        categorical_cols = ["home_team", "away_team", "status", "season"]
        label_encoders = {col: LabelEncoder() for col in categorical_cols}

        for col in categorical_cols:
            df[col] = label_encoders[col].fit_transform(df[col])

        # Convert match_day to numeric if not already
        df['match_day'] = pd.to_datetime(df['match_day'], format='%d.%m.%Y', errors='coerce')
        df['match_day'] = df['match_day'].fillna(0).infer_objects(copy=False)

        # Prepare features and target
        X = df.drop(columns=["match_outcome"])
        y = df["match_outcome"]

        # Scale numerical features
        scaler = StandardScaler()
        numerical_cols = ['match_day', 'home_team_avg_goals', 'home_team_def_strength',
                          'home_team_win_ratio', 'away_team_avg_goals',
                          'away_team_def_strength', 'away_team_win_ratio']
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestClassifier(n_estimators=100,
                                       max_depth=10,
                                       min_samples_split=10,
                                       class_weight='balanced',
                                       random_state=42)
        model.fit(X_train, y_train)

        # Evaluate model
        y_pred = model.predict(X_test)
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
        


        return model, label_encoders, scaler
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def main():
    try:
        raw_data = fetch_the_data()
        if raw_data is None:
            return

        df, team_stats = clean_csv(raw_data, 'cleaned.csv')
        if df is None:
            return

        # Train model
        model, encoders, scaler = train_model(df)
        if model is None:
            return


    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()