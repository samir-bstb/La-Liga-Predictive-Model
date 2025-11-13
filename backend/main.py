from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from .model import predict_match, simulate_season_with_predict_match
import pickle
import os 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # restraint to my own domain
    allow_methods=["*"],
    allow_headers=["*"],
)

class MatchRequest(BaseModel):
    home: str
    away: str

@app.post("/predict")
def predict_match_endpoint(request: MatchRequest):
    result = predict_match(request.home, request.away)
    return result

@app.get("/predict-season")
def predict_season_endpoint():
    with open("model.pkl", "rb") as f:
        model, scaler, features, team_latest, data = pickle.load(f)

    season_df = data[data["season"] == "2024-25"].copy()
    if season_df.empty:
        return {"error": "There is no data from the 24/25 season"}
    
    avg_home_goals = data['FTHG'].mean()
    avg_away_goals = data['FTAG'].mean()

    table = simulate_season_with_predict_match(
        season_df=season_df,
        model=model,
        features=features,
        lambda_h=avg_home_goals,
        lambda_a=avg_away_goals,
        scaler=scaler
    )

    result = []
    position = 1
    for team, stats in table.items():
        result.append({
            "position": position,
            "team": team,
            "points": stats["points"],
            "gf": stats["GF"],
            "ga": stats["GA"],
            "gd": stats["GD"]
        })
        position += 1

    return {"season": "2024-25", "table": result}

# For production (Render uses PORT env var)
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))  # Render asigns PORT
    uvicorn.run(app, host="0.0.0.0", port=port)  # accesible fromthe outside
