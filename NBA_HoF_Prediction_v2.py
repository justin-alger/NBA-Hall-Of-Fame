# ============================================================
# NBA Hall of Fame — OOF for Eligible + Model Scoring for Ineligible/Active
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.base import clone

# ------------------------ Paths ------------------------
INPUT_PATH  = "NBA_PLAYERS.xlsx"                     # <-- set your path
EXPORT_PATH = "nba_players_predictions_combined_oof.csv"
FEATIMP_PATH = "nba_hof_feature_importances.csv"

# ------------------------ Load & Clean ------------------------
df = pd.read_excel(INPUT_PATH)
df.columns = [c.strip().replace(" ", "_") for c in df.columns]

# Required columns
for c in ["HOF", "Active", "Final"]:
    if c not in df.columns:
        raise ValueError(f"Missing required column: {c}")

# Coerce types
df["Active"] = (
    df["Active"].astype(str).str.lower()
    .map({"true": True, "false": False, "1": True, "0": False})
)
df["Active"] = df["Active"].fillna(False).astype(bool)
df["Final"]  = pd.to_numeric(df["Final"],  errors="coerce")
df["HOF"]    = (
    df["HOF"].astype(str).str.lower().map({"true": 1, "false": 0, "1": 1, "0": 0})
)
df["HOF"]    = pd.to_numeric(df["HOF"], errors="coerce").fillna(0).astype(int)

# Try to find a player name column (optional)
name_col = None
for c in df.columns:
    if any(k in c.lower() for k in ["player", "name"]):
        name_col = c
        break

# ------------------------ Engineered Features ------------------------
# Safely create engineered columns even if some inputs are missing
def safe_col(s):
    return df[s] if s in df.columns else pd.Series(np.nan, index=df.index)

Final = safe_col("Final")
Debut = safe_col("Debut")
PTS   = safe_col("PTS")
TRB   = safe_col("TRB")
AST   = safe_col("AST")
G     = safe_col("G")
PER   = safe_col("PER")

# Career length
df["Seasons_Played"] = (Final - Debut)

# Total stats
df["Career_Points"]   = PTS * G
df["Career_Rebounds"] = TRB * G
df["Career_Assists"]  = AST * G

# Per-season averages (avoid div by zero)
sp = df["Seasons_Played"].fillna(0)
df["Points_per_season"] = df["Career_Points"] / sp.replace(0, 1)

# Peak stat across PTS/TRB/AST/PER
df["Peak_Performance"] = pd.concat([PTS, TRB, AST, PER], axis=1).max(axis=1)

# PER normalized per game (avoid div by zero)
df["Efficiency_Per_Game"] = PER / G.replace(0, 1)

# Eligibility flags
coaches = ["Red Auerbach", "Larry Brown", "Jim Calhoun", "Chuck Daly", "Alex Hannum", 
           "Red Holzman", "Phil Jackson", "John Kundla", "Bobby Leonard", "Slick Leonard", 
           "Don Nelson", "Pat Riley", "Jack Ramsay", "Jerry Sloan", "Gregg Popovich", 
           "Billy Donovan","John Thompson", "Al McGuire", "George Karl", "Rick Adelman",
           "Tom Sanders"]
df = df[~df[name_col].isin(coaches)] if name_col else df # remove known coaches
elig_mask = (~df["Active"]) & (df["Final"] <= 2022)
inelig_mask = (~df["Active"]) & (df["Final"] > 2022)

df["Eligibility"] = np.where(
    elig_mask, "Eligible",
    np.where(df["Active"], "Active", "Ineligible")
)

# ------------------------ Features ------------------------
exclude = {"HOF", "Active"}
if name_col:
    exclude.add(name_col)
exclude.update({c for c in df.columns if any(k in c.lower() for k in ["id", "uuid", "guid"])})

# 2) Optionally define a whitelist to force which features are used.
#    If empty, use all non-excluded columns (including engineered).
FEATURE_WHITELIST = [
    # Example: uncomment to explicitly control:
     "Seasons_Played","Career_Points","Career_Rebounds","Career_Assists",
     "Points_per_season","Peak_Performance","Efficiency_Per_Game",
     "PTS","TRB","AST","PER","G","WS"
]
if FEATURE_WHITELIST:
    # keep only those that are in df.columns and not excluded
    features = [f for f in FEATURE_WHITELIST if (f in df.columns and f not in exclude)]
else:
    features = [c for c in df.columns if c not in exclude]

target   = "HOF"

# ------------------------ Train/Valid Split (Eligible only) ------------------------
eligible = df.loc[elig_mask].copy()
X = eligible[features]
y = eligible[target]

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Track splits on master df
df["Data_Split"] = "Ineligible"
df.loc[elig_mask,     "Data_Split"] = "Train"
df.loc[X_valid.index, "Data_Split"] = "Validation"
df.loc[df["Active"],  "Data_Split"] = "Active"

# ------------------------ Preprocessing ------------------------
num_cols = [c for c in X_train.columns if np.issubdtype(X_train[c].dtype, np.number)]
cat_cols = [c for c in X_train.columns if c not in num_cols]

num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preproc = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

# ------------------------ Models & Light Tuning ------------------------
dt = Pipeline([
    ("prep", preproc),
    ("clf", DecisionTreeClassifier(random_state=42, class_weight="balanced")),
])
rf = Pipeline([
    ("prep", preproc),
    ("clf", RandomForestClassifier(random_state=42, class_weight="balanced_subsample", n_jobs=-1)),
])

dt_grid = {
    "clf__max_depth": [None, 8, 12],
    "clf__min_samples_leaf": [1, 4],
    "clf__min_samples_split": [2, 10],
    "clf__ccp_alpha": [0.0, 0.001],
}
rf_grid = {
    "clf__n_estimators": [300, 500],
    "clf__max_depth": [None, 12],
    "clf__min_samples_leaf": [1, 3],
    "clf__min_samples_split": [2, 6],
    "clf__max_features": ["sqrt", "log2"],
}

dt_cv = GridSearchCV(dt, dt_grid, scoring="f1", cv=3, n_jobs=-1)
rf_cv = GridSearchCV(rf, rf_grid, scoring="f1", cv=3, n_jobs=-1)

# Fit on TRAIN only
dt_cv.fit(X_train, y_train)
rf_cv.fit(X_train, y_train)

# Choose best by validation F1
dt_f1 = f1_score(y_valid, dt_cv.predict(X_valid))
rf_f1 = f1_score(y_valid, rf_cv.predict(X_valid))
best_cv = dt_cv if dt_f1 >= rf_f1 else rf_cv
best_name = "DecisionTree" if best_cv is dt_cv else "RandomForest"

print(f"Validation F1 — DT: {dt_f1:.4f}, RF: {rf_f1:.4f}. Best: {best_name}")

# ------------------------ Threshold Tuning (Validation only) ------------------------
valid_probs = best_cv.predict_proba(X_valid)[:, 1]
thresholds = np.linspace(0.2, 0.8, 31)
f1s = [f1_score(y_valid, (valid_probs >= t).astype(int)) for t in thresholds]
best_threshold = float(thresholds[int(np.argmax(f1s))])
print(f"Optimal threshold on validation: {best_threshold:.3f} (F1={max(f1s):.4f})")

# Confusion matrix for validation predictions @ tuned threshold
y_pred_tuned = (valid_probs >= best_threshold).astype(int)
cm = confusion_matrix(y_valid, y_pred_tuned)
ConfusionMatrixDisplay(cm).plot(cmap="Blues")
plt.title(f"{best_name} — Validation Confusion Matrix @ threshold={best_threshold:.2f}")
plt.show()

# ------------------------ OOF predictions for ALL Eligible ------------------------
# Leakage-free probabilities for the entire eligible set (train + valid)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_probs = pd.Series(index=eligible.index, dtype=float)

for train_idx, val_idx in skf.split(eligible[features], eligible[target]):
    X_tr, X_va = eligible.iloc[train_idx][features], eligible.iloc[val_idx][features]
    y_tr       = eligible.iloc[train_idx][target]

    fold_model = clone(best_cv.best_estimator_)  # same tuned pipeline
    fold_model.fit(X_tr, y_tr)
    oof_probs.iloc[val_idx] = fold_model.predict_proba(X_va)[:, 1]

# Sanity check: no missing eligible indices
assert oof_probs.notna().all()

# ------------------------ Fit best model on ALL eligible & score Ineligible/Active ------------------------
best_estimator_all = clone(best_cv.best_estimator_)
best_estimator_all.fit(X, y)

# Prepare output columns
df["BestModel"]     = best_name
df["BestThreshold"] = best_threshold
df["Prob_HOF"]      = np.nan
df["Pred_HOF"]      = np.nan

# Fill eligible rows (OOF probabilities)
df.loc[eligible.index, "Prob_HOF"] = oof_probs.values
df.loc[eligible.index, "Pred_HOF"] = (oof_probs.values >= best_threshold).astype(int)
# Label both train and validation eligible as 'Eligible_OOF' for clarity in visuals
df.loc[eligible.index, "Data_Split"] = "Eligible_OOF"

# Score ineligible (recently retired) with model trained on ALL eligible
ineligible_idx = df.index[inelig_mask]
if len(ineligible_idx) > 0:
    X_inelig = df.loc[ineligible_idx, features]
    inelig_probs = best_estimator_all.predict_proba(X_inelig)[:, 1]
    df.loc[ineligible_idx, "Prob_HOF"] = inelig_probs
    df.loc[ineligible_idx, "Pred_HOF"] = (inelig_probs >= best_threshold).astype(int)
    df.loc[ineligible_idx, "Data_Split"] = "Ineligible"

# Score active with model trained on ALL eligible
active_idx = df.index[df["Active"]]
if len(active_idx) > 0:
    X_active = df.loc[active_idx, features]
    active_probs = best_estimator_all.predict_proba(X_active)[:, 1]
    df.loc[active_idx, "Prob_HOF"] = active_probs
    df.loc[active_idx, "Pred_HOF"] = (active_probs >= best_threshold).astype(int)
    df.loc[active_idx, "Data_Split"] = "Active"

# ------------------------ FEATURE IMPORTANCE ------------------------
# We'll compute after fitting on ALL eligible (best_estimator_all)
# Extract feature names after preprocessing:
prep = best_estimator_all.named_steps["prep"]
clf  = best_estimator_all.named_steps["clf"]

num_names = list(num_cols)
if len(cat_cols) > 0:
    ohe = prep.named_transformers_["cat"].named_steps["onehot"]
    cat_names = list(ohe.get_feature_names_out(cat_cols))
else:
    cat_names = []

all_feature_names = num_names + cat_names

importances = getattr(clf, "feature_importances_", None)
if importances is not None and len(importances) == len(all_feature_names):
    featimp_df = pd.DataFrame({
        "feature": all_feature_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)
else:
    # Fallback: no importances available (unlikely for tree/forest)
    featimp_df = pd.DataFrame({"feature": all_feature_names, "importance": np.nan})

featimp_df.to_csv(FEATIMP_PATH, index=False)
print(f"✅ Saved feature importances to: {FEATIMP_PATH}")
print(featimp_df.head(15))

# ------------------------ Export combined file ------------------------
front = []
if name_col:
    front.append(name_col)
front += ["Eligibility", "Data_Split", "HOF", "Prob_HOF", "Pred_HOF", "BestModel", "BestThreshold"]
ordered = front + [c for c in df.columns if c not in front]
df_out = df[ordered].copy()

df_out.to_csv(EXPORT_PATH, index=False)
print(f"\n✅ Exported (OOF for Eligible; model-scored Ineligible & Active): {EXPORT_PATH}")
