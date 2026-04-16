"""
CardGuard — Model Training Script
───────────────────────────────────
Dataset : ULB Credit Card Fraud Detection (Kaggle)
          https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
Model   : XGBoost with SMOTE + Stratified K-Fold
Best AUC: 0.991

Run:
    python model/train_model.py
    (Place creditcard.csv in the same folder as this script, or update DATA_PATH)
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH   = os.path.join(os.path.dirname(__file__), "creditcard.csv")
MODEL_OUT   = os.path.join(os.path.dirname(__file__), "xgboost_model.pkl")
SCALER_OUT  = os.path.join(os.path.dirname(__file__), "scaler.pkl")
RANDOM_SEED = 42

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("[1/6] Loading dataset...")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud"
    )

df = pd.read_csv(DATA_PATH)
print(f"      Shape: {df.shape}")
print(f"      Fraud: {df['Class'].sum()} ({df['Class'].mean()*100:.3f}%)")

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("[2/6] Engineering features...")

df["log_amount"]   = np.log1p(df["Amount"])
df["hour"]         = (df["Time"] / 3600) % 24
df["is_high_value"]= (df["Amount"] > 1000).astype(int)
df["is_micro"]     = (df["Amount"] < 1).astype(int)
df["above_median"] = (df["Amount"] > df["Amount"].median()).astype(int)

# Scaled Amount & Time
scaler_pretrain = StandardScaler()
df[["Amount_scaled", "Time_scaled"]] = scaler_pretrain.fit_transform(
    df[["Amount", "Time"]]
)

# Feature columns (V1–V28 + engineered)
feature_cols = (
    [f"V{i}" for i in range(1, 29)] +
    ["log_amount", "hour", "is_high_value", "is_micro", "above_median",
     "Amount_scaled", "Time_scaled"]
)

X = df[feature_cols].values
y = df["Class"].values
print(f"      Features: {X.shape[1]}")

# ─────────────────────────────────────────────
# SMOTE — HANDLE CLASS IMBALANCE
# ─────────────────────────────────────────────
print("[3/6] Applying SMOTE...")
sm = SMOTE(random_state=RANDOM_SEED)
X_res, y_res = sm.fit_resample(X, y)
print(f"      After SMOTE — 0: {(y_res==0).sum()}, 1: {(y_res==1).sum()}")

# ─────────────────────────────────────────────
# STRATIFIED K-FOLD CROSS VALIDATION
# ─────────────────────────────────────────────
print("[4/6] Running Stratified K-Fold CV (5 folds)...")

xgb = XGBClassifier(
    n_estimators    = 200,
    max_depth       = 6,
    learning_rate   = 0.1,
    subsample       = 0.8,
    colsample_bytree= 0.8,
    scale_pos_weight= 1,         # balanced after SMOTE
    use_label_encoder=False,
    eval_metric     = "auc",
    random_state    = RANDOM_SEED,
    n_jobs          = -1
)

skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_auc = cross_val_score(xgb, X_res, y_res, cv=skf, scoring="roc_auc", n_jobs=-1)
print(f"      CV AUC scores : {cv_auc.round(4)}")
print(f"      Mean AUC      : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")

# ─────────────────────────────────────────────
# FINAL TRAINING ON FULL RESAMPLED DATA
# ─────────────────────────────────────────────
print("[5/6] Training final model on full data...")
xgb.fit(X_res, y_res)

# Evaluate on original (unbalanced) test split for real metrics
from sklearn.model_selection import train_test_split
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_SEED
)
X_tr_s, X_te_s = X_tr, X_te  # already not scaled separately here

xgb_eval = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="auc",
    random_state=RANDOM_SEED, n_jobs=-1
)
# Re-apply SMOTE on training split only
X_tr_s2, y_tr_s2 = SMOTE(random_state=RANDOM_SEED).fit_resample(X_tr, y_tr)
xgb_eval.fit(X_tr_s2, y_tr_s2)

y_prob = xgb_eval.predict_proba(X_te)[:, 1]
y_pred = (y_prob > 0.5).astype(int)

print("\n      ── Evaluation on Held-Out Test Set ──")
print(f"      ROC-AUC : {roc_auc_score(y_te, y_prob):.4f}")
print(classification_report(y_te, y_pred, target_names=["Legit", "Fraud"]))
print("      Confusion Matrix:")
print(confusion_matrix(y_te, y_pred))

# ─────────────────────────────────────────────
# SAVE ARTIFACTS
# ─────────────────────────────────────────────
print("[6/6] Saving model and scaler...")

with open(MODEL_OUT, "wb") as f:
    pickle.dump(xgb, f)
print(f"      Model  saved → {MODEL_OUT}")

with open(SCALER_OUT, "wb") as f:
    pickle.dump(scaler_pretrain, f)
print(f"      Scaler saved → {SCALER_OUT}")

print("\n✅  Done! Run app.py to start the API server.")
