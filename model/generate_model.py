"""
CardGuard — Dummy Model Generator
Generates xgboost_model.pkl + scaler.pkl using synthetic data.
This is used when the real Kaggle dataset is not available (for deployment).
The model structure is identical to the real trained model.
"""
import os, pickle, numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

print("[1/3] Generating synthetic training data (35 features)...")
N = 10000
# Simulate V1-V28 as normal distributions
X_legit = np.random.randn(int(N*0.998), 35)
# Fraud transactions have different distribution
X_fraud  = np.random.randn(int(N*0.002), 35) * 1.5 + 0.5
X = np.vstack([X_legit, X_fraud])
y = np.array([0]*len(X_legit) + [1]*len(X_fraud))

# Shuffle
idx = np.random.permutation(len(X))
X, y = X[idx], y[idx]

print(f"      Data shape: {X.shape}, Fraud: {y.sum()}")

print("[2/3] Training XGBoost model...")
model = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="auc",
    random_state=RANDOM_SEED, n_jobs=-1
)
model.fit(X, y)

scaler = StandardScaler()
scaler.fit(X)

print("[3/3] Saving model artifacts...")
out_dir = os.path.dirname(__file__)
with open(os.path.join(out_dir, "xgboost_model.pkl"), "wb") as f:
    pickle.dump(model, f)
with open(os.path.join(out_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)

print("✅  xgboost_model.pkl and scaler.pkl saved!")
