La Liga Winner Prediction Model for the 2024/2025 Season

Overview
Thsi project implements a machine learning model to predict the 
winner of La Liga for the 2024/2025 season. The 
approach combines XGBoost dor multi-class classification to predict 
match outcomes (home win, draw, away win) and a Poisson distribution
to simulate realistic goal counts in prediction functions.
The model levarages historica match data and engineered features to 
provide robust predictions and simulate season outcomes.

Methodology
The prediction model consisits of two primary components:
1. XGBoost for multi-class cassification:
    - Purpose: Predicts match outcomes (Home win, Draw, Away win) based
      on historical and team performance features.
    - Algorithm: XGBoost, configured with the multi:softprob objective to
      output probabilities for each outcome class.

    Training Features:
      - home_recent_gd: Average goal difference for the home team over the last 12 matches.
      - away_recent_gd: Average goal difference for the away team over the last 12 matches.
      - home_recent_points: Average points earned by the home team over the last 5 matches.
      - away_recent_points: Average points earned by the away team over the last 5 matches.
      - h2h_H, h2h_D, h2h_A: Head-to-head statistics representing the proportion of home wins, draws, and away wins, respectively, in historical matches between the two teams. For training, all prior head-to-head matches are considered; for predictions, the last 10 matches are used.

    Data Preprocessing: Features are normalized using StandardScaler to
    ensure stable model training. Custom class weights are applied to address
    class imbalance, particularly for draws.

2. Poisson distribution for goal simulation
  - Purpose: Generates realistic goal counts for simulated matches in season
    outcome predictions.
  - Implementation: The Poisson distribution is used to sample home and away
    goals based on league-wide average goal rates (lambda_h for home goals,
    lambda_a for away goals). Goals are sampled conditionally to align with
    the predicted match outcome from XGBoost (e.g., home goals > away goals
    for a home win).
  - Application: Used in the simulate_season_realistic and simulate_season_with_predict_match
    functions to simulate full seasons and compute team standings.

Data
  - Source: Historical La Liga match data, loaded from CSV files covering 
  multiple seasons. Retrieved from https://www.football-data.co.uk/

  - Feature Engineering: Includes rolling averages for goal differences and 
  points, as well as head-to-head statistics, to capture team form and 
  historical performance.

  - Data Splitting:
    - Training: Seasons 2019/20 to 2022/23.
    - Validation: Season 2023/24.
    - Testing: Season 2024/25.

Model evaluation
The model is evaluated using:
  - Accuracy: Proportion of correctly predicted match outcomes.
  - Log Loss: Measures the quality of predicted probabilities.
  - Confusion Matrix: Analyzes prediction performance across outcome classes (H, D, A).

Simulation
The model simulates the 2024/2025 La Liga season by:
  - Predicting match outcomes using XGBoost.
  - Simulating goal counts with the Poisson distribution.
  - Aggregating results to compute team standings based on points, goals for
    (GF), goals against (GA), and goal difference (GD).



Dependencies
  - Python 3.0
  - Libraries: pandas, numpy, xgboost, scikit-learn, matplotlib



Dependencies
  - Python 3.0
  - Libraries: pandas, numpy, xgboost, scikit-learn, matplotlib

Usage
  - Load and preprocess historical La Liga data.
  - Train the XGBoost model on the engineered features.
  - Use the trained model to predict individual match outcomes or simulate the entire season.
  - Evaluate model performance using the provided metrics.

For further details, refer to the source code and documentation within the repository.
