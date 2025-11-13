import pandas as pd
import numpy as np
import xgboost as xgb
import requests
from io import StringIO
from sklearn.preprocessing import StandardScaler
import pickle
import os

season_urls = {
    '2019-20': 'https://www.football-data.co.uk/mmz4281/1920/SP1.csv',
    '2020-21': 'https://www.football-data.co.uk/mmz4281/2021/SP1.csv',
    '2021-22': 'https://www.football-data.co.uk/mmz4281/2122/SP1.csv',
    '2022-23': 'https://www.football-data.co.uk/mmz4281/2223/SP1.csv',
    '2023-24': 'https://www.football-data.co.uk/mmz4281/2324/SP1.csv',
    '2024-25': 'https://www.football-data.co.uk/mmz4281/2425/SP1.csv'
}


print("Loading historicla data...")
dfs = []
for season, url in season_urls.items():
    print(f"  → {season}")
    r = requests.get(url)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df["season"] = season
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)
data['Date'] = pd.to_datetime(data['Date'], dayfirst=True, errors='coerce')
data = data.sort_values('Date').reset_index(drop=True)

def build_features(df):
    df = df.copy()
    df["goal_diff"] = df["FTHG"] - df["FTAG"]
    df["result"] = np.where(df["goal_diff"] > 0, "H", np.where(df["goal_diff"] < 0, "A", "D"))
    df["home_points"] = np.where(df["result"] == "H", 3, np.where(df["result"] == "D", 1, 0))
    df["away_points"] = np.where(df["result"] == "A", 3, np.where(df["result"] == "D", 1, 0))

    home_df = df[['Date', 'HomeTeam', 'AwayTeam', 'goal_diff', 'home_points']].rename(
        columns={'HomeTeam': 'team', 'AwayTeam': 'opponent', 'home_points': 'points'})
    home_df['gd'] = df['goal_diff']
    home_df['venue'] = 'home'

    away_df = df[['Date', 'AwayTeam', 'HomeTeam', 'goal_diff', 'away_points']].rename(
        columns={'AwayTeam': 'team', 'HomeTeam': 'opponent', 'away_points': 'points'})
    away_df['gd'] = -df['goal_diff']
    away_df['venue'] = 'away'

    team_matches = pd.concat([home_df, away_df])
    team_matches = team_matches.sort_values(['team', 'Date']).reset_index(drop=True)

    team_matches['recent_gd'] = team_matches.groupby('team')['gd'].transform(
        lambda x: x.rolling(12, min_periods=1).mean().shift(1))
    team_matches['recent_points'] = team_matches.groupby('team')['points'].transform(
        lambda x: x.rolling(5, min_periods=1).mean().shift(1))

    team_matches['recent_gd'] = team_matches['recent_gd'].fillna(0)
    team_matches['recent_points'] = team_matches['recent_points'].fillna(0)

    home_features = team_matches[team_matches['venue'] == 'home'][['Date', 'team', 'recent_gd', 'recent_points']].rename(
        columns={'recent_gd': 'home_recent_gd', 'recent_points': 'home_recent_points'})
    df = df.merge(home_features, left_on=['Date', 'HomeTeam'], right_on=['Date', 'team'], how='left').drop('team', axis=1)

    away_features = team_matches[team_matches['venue'] == 'away'][['Date', 'team', 'recent_gd', 'recent_points']].rename(
        columns={'recent_gd': 'away_recent_gd', 'recent_points': 'away_recent_points'})
    df = df.merge(away_features, left_on=['Date', 'AwayTeam'], right_on=['Date', 'team'], how='left').drop('team', axis=1)

    df['h2h_H'] = 0.0
    df['h2h_D'] = 0.0
    df['h2h_A'] = 0.0

    for idx in range(len(df)):
        row = df.iloc[idx]
        home, away = row['HomeTeam'], row['AwayTeam']
        past = df.iloc[:idx]
        mask1 = (past["HomeTeam"] == home) & (past["AwayTeam"] == away)
        mask2 = (past["HomeTeam"] == away) & (past["AwayTeam"] == home)
        matches = past[mask1 | mask2]
        total = len(matches)
        if total == 0:
            df.at[idx, 'h2h_H'] = 0.333
            df.at[idx, 'h2h_D'] = 0.333
            df.at[idx, 'h2h_A'] = 0.334
            continue
        h_wins = ((matches["HomeTeam"] == home) & (matches["FTHG"] > matches["FTAG"])).sum() + \
                 ((matches["AwayTeam"] == home) & (matches["FTAG"] > matches["FTHG"])).sum()
        draws = (matches["FTHG"] == matches["FTAG"]).sum()
        a_wins = total - h_wins - draws
        df.at[idx, 'h2h_H'] = h_wins / total
        df.at[idx, 'h2h_D'] = draws / total
        df.at[idx, 'h2h_A'] = a_wins / total

    df['h2h_H'] *= 1.2
    df['h2h_D'] *= 1.1
    df['h2h_A'] *= 1.3

    features = ["home_recent_gd", "away_recent_gd", "home_recent_points", "away_recent_points", "h2h_H", "h2h_D", "h2h_A"]
    return df, features, team_matches

def compute_h2h_features(df, home, away, n_matches=10):
    mask1 = (df["HomeTeam"] == home) & (df["AwayTeam"] == away)
    mask2 = (df["HomeTeam"] == away) & (df["AwayTeam"] == home)
    matches = df[mask1 | mask2].sort_values("Date", ascending=False).head(n_matches)
    total = len(matches)
    if total == 0:
        return {"h2h_H": 0.333, "h2h_D": 0.333, "h2h_A": 0.334}
    h_wins = ((matches["HomeTeam"] == home) & (matches["FTHG"] > matches["FTAG"])).sum() + \
             ((matches["AwayTeam"] == home) & (matches["FTAG"] > matches["FTHG"])).sum()
    draws = (matches["FTHG"] == matches["FTAG"]).sum()
    a_wins = total - h_wins - draws
    return {"h2h_H": h_wins / total, "h2h_D": draws / total, "h2h_A": a_wins / total}

# Tarin model
data, features, team_matches = build_features(data)
team_latest = team_matches.groupby('team')[['recent_gd', 'recent_points']].last()

train = data[data["season"].isin(["2019-20", "2020-21", "2021-22", "2022-23"])]
X_train, y_train = train[features], train["result"]
label_map = {"H": 0, "D": 1, "A": 2}
y_train_num = y_train.map(label_map)

custom_weights = {0: 1.3, 1: 0.75, 2: 1.0}
sample_weights = [custom_weights[label] for label in y_train_num]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

dtrain = xgb.DMatrix(X_train_scaled, label=y_train_num, weight=sample_weights)

params = {
    "objective": "multi:softprob",
    "num_class": 3,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "max_depth": 3,
    "eta": 0.03,
    "lambda": 8.0,
    "alpha": 4.0,
    "subsample": 0.8,
    "colsample_bytree": 0.8
}

model = xgb.train(params, dtrain, num_boost_round=250)

# Save model to train it once
with open("model.pkl", "wb") as f:
    pickle.dump((model, scaler, features, team_latest, data), f)

print("Model trained and saved.")

# Predictive function
def predict_match(home, away):
    with open("model.pkl", "rb") as f:
        model, scaler, features, team_latest, data = pickle.load(f)

    home_recent_gd = team_latest.loc[home, 'recent_gd'] if home in team_latest.index else 0
    home_recent_points = team_latest.loc[home, 'recent_points'] if home in team_latest.index else 0
    away_recent_gd = team_latest.loc[away, 'recent_gd'] if away in team_latest.index else 0
    away_recent_points = team_latest.loc[away, 'recent_points'] if away in team_latest.index else 0

    h2h = compute_h2h_features(data, home, away, n_matches=10)
    h2h["h2h_H"] *= 1.2
    h2h["h2h_D"] *= 1.1
    h2h["h2h_A"] *= 1.3

    row = {
        "home_recent_gd": home_recent_gd,
        "away_recent_gd": away_recent_gd,
        "home_recent_points": home_recent_points,
        "away_recent_points": away_recent_points,
        "h2h_H": h2h["h2h_H"],
        "h2h_D": h2h["h2h_D"],
        "h2h_A": h2h["h2h_A"]
    }

    X = pd.DataFrame([row])[features]
    X_scaled = scaler.transform(X)
    dX = xgb.DMatrix(X_scaled)
    probs = model.predict(dX)[0]

    return {
        "H": float(round(probs[0] * 100, 1)),
        "D": float(round(probs[1] * 100, 1)),
        "A": float(round(probs[2] * 100, 1))
    }

def simulate_season_with_predict_match(season_df, model, features, lambda_h, lambda_a, scaler):
    season_df = season_df.sort_values('Date').reset_index(drop=True)
    teams = sorted(set(season_df["HomeTeam"]).union(set(season_df["AwayTeam"])))
    table = {t: {"points": 0, "GF": 0, "GA": 0, "GD": 0} for t in teams}

    for _, row in season_df.iterrows():
        home, away = row["HomeTeam"], row["AwayTeam"]
        probs = predict_match(home, away)  # devuelve porcentajes

        p = np.array([probs["H"], probs["D"], probs["A"]]) / 100
        p = p / p.sum()

        outcome = np.random.choice(["H", "D", "A"], p=p)

        while True:
            home_goals = np.random.poisson(lambda_h)
            away_goals = np.random.poisson(lambda_a)
            if (outcome == "H" and home_goals > away_goals) or \
               (outcome == "D" and home_goals == away_goals) or \
               (outcome == "A" and home_goals < away_goals):
                break

        table[home]["GF"] += home_goals
        table[home]["GA"] += away_goals
        table[away]["GF"] += away_goals
        table[away]["GA"] += home_goals

        if outcome == "H":
            table[home]["points"] += 3
        elif outcome == "A":
            table[away]["points"] += 3
        else:
            table[home]["points"] += 1
            table[away]["points"] += 1

    for t in teams:
        table[t]["GD"] = table[t]["GF"] - table[t]["GA"]

    table_sorted = dict(sorted(table.items(), key=lambda x: (x[1]["points"], x[1]["GD"]), reverse=True))
    return table_sorted

#tpye in terminal: uvicorn main:app --reload      
