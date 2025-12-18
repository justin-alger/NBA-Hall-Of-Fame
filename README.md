# Predicting the NBA Hall of Famers - With Data
A machine learning sports analytics project using NBA player data and Hall of Fame inductions to predict future NBA Hall of Famers.
## 🎯 Overview
This project explores whether machine learning can predict which NBA players will make the Hall of Fame, based on their career stats, advanced metrics, and efficiency measures.
We trained and validated models on retired eligible players (≥3 years since final season), engineered new career-level features, and used Random Forests and Decision Trees to predict Hall of Fame probability for active and recently retired players. 

The best model achieved an F1 score of 0.69 through out-of-fold validation.
## 🧩 Project Goals
- Predict which NBA players are likely to be inducted into the Hall of Fame
- Identify the most influential factors driving induction probability
- Visualize the results in an interactive Tableau dashboard
- Create an interpretable machine learning workflow
## 🧮 Feature Engineering
To go beyond box-score stats, several engineered features were created:
- Seasons_Played:	Career length (Final - Debut)
- Career_Points, Career_Rebounds, Career_Assists:	Total lifetime contributions (PTS * G, etc.)
- Points_per_season:	Points normalized by career length
- Peak_Performance:	Highest of PTS, TRB, AST, PER
- Efficiency_Per_Game:	PER normalized by games played
- Win_Shares:	Estimated wins contributed to team success (top predictor)
## ⚙️ Modeling Process
1️⃣ Data Preparation
- Filtered to include only eligible retired players for training.
- Active and recently retired players used only for scoring.
- Had to filter out former players that are in the Hall of Fame for their coaching, not player, contributions

2️⃣ Model Training
- Algorithms: Decision Tree & Random Forest
- Preprocessing: Median imputation + One-Hot Encoding
- Cross-validation: 5-fold Stratified CV (OOF predictions)
- Metric: F1 Score (balances precision and recall)

3️⃣ Validation
- Tuned probability threshold for best F1
- Produced confusion matrix, feature importances, and OOF probabilities
## 📈 Key Results
- Best Model:	Random Forest
- Validation F1 Score:	0.69
- Precision / Recall:	0.74 / 0.64
- Cross-Validation:	5-fold OOF predictions (no leakage)
## 🏆 Top Predictive Features
1️⃣ Win Shares
2️⃣ Career Points
3️⃣ Points per Game
4️⃣ Career Assists
5️⃣ Peak Performance

### 🧠 Why Win Shares?
It quantifies total impact on team wins — blending offense, defense, and efficiency.
While PPG tells you what a player did, Win Shares tells you how much it mattered.

## 🧠 Understanding the F1 Score

The F1 Score balances precision (accuracy of positive predictions) and recall (how many positives you find).
F1=2 × (Precision×Recall / Precision+Recall)

Precision: How many predicted Hall of Famers actually were (105 / 141 = 74.4%)

Recall: How many true Hall of Famers we caught (105 / 163 = 64.4%)

A model with F1 = 0.69 correctly identifies about 7 of 10 inductees while minimizing false positives.

## 📊 Tableau Dashboard to Visualize

Explore the interactive Tableau dashboard to:
- See predicted Hall of Fame probabilities
- Compare active vs retired players
- Filter by career metrics and rankings

🔗 View the Dashboard on Tableau Public: https://public.tableau.com/app/profile/justin.alger/viz/NBAHallofFamePredictions/HallofFame
