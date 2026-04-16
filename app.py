"""
CardGuard — Credit Card Fraud Detection
Flask Backend API
Author  : karritejaswini2003 (Roll No. 49, MCA — Aditya Degree & PG College)
Guide   : Sai Vardhan Sir
Dataset : ULB Credit Card Fraud Dataset (Kaggle)
Model   : XGBoost (AUC 0.991) with SMOTE + Stratified K-Fold
"""

import os
import pickle
import threading
import smtplib
from email.mime.text import MIMEText
from datetime import datetime

import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# ─────────────────────────────────────────────
# APP SETUP
# ─────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# LOAD MODEL & SCALER
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE_DIR, "model", "xgboost_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")

model  = None
scaler = None

def load_artifacts():
    global model, scaler
    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
        print(f"[✓] Model loaded from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"[!] Model not found at {MODEL_PATH}.")
    except Exception as e:
        print(f"[!] Error loading model: {e}")

    try:
        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)
        print(f"[✓] Scaler loaded from {SCALER_PATH}")
    except FileNotFoundError:
        print(f"[!] Scaler not found at {SCALER_PATH}.")
    except Exception as e:
        print(f"[!] Error loading scaler: {e}")

load_artifacts()

# ─────────────────────────────────────────────
# EMAIL ALERT (Gmail SMTP)
# ─────────────────────────────────────────────
SENDER_EMAIL    = os.getenv("ALERT_EMAIL",    "")
SENDER_PASSWORD = os.getenv("ALERT_PASSWORD", "")

def send_email_alert(to_email, txn_id, amount, probability, risk):
    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("[!] Email credentials not set. Skipping email alert.")
        return
    try:
        subject = f"⚠ Fraud Alert — CardGuard | {txn_id}"
        body = (
            f"CardGuard Fraud Detection System\n"
            f"─────────────────────────────────\n"
            f"Transaction ID   : {txn_id}\n"
            f"Amount           : ₹{amount:.2f}\n"
            f"Fraud Probability: {probability*100:.1f}%\n"
            f"Risk Level       : {risk}\n"
            f"Detected At      : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            f"ACTION REQUIRED: Please verify this transaction immediately.\n"
            f"If this was not you, contact your bank to block the card.\n\n"
            f"— CardGuard Automated Alert"
        )
        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"]    = SENDER_EMAIL
        msg["To"]      = to_email
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, to_email, msg.as_string())
        print(f"[✓] Email alert sent to {to_email}")
    except Exception as e:
        print(f"[!] Email failed: {e}")

# ─────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────
def build_features(data):
    raw_features = data.get("features", [0] * 35)
    if len(raw_features) < 35:
        raw_features = raw_features + [0.0] * (35 - len(raw_features))
    raw_features = raw_features[:35]
    return np.array(raw_features, dtype=np.float32).reshape(1, -1)

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({
        "status"       : "ok",
        "model_loaded" : model is not None,
        "scaler_loaded": scaler is not None,
        "timestamp"    : datetime.now().isoformat()
    }), 200

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    X = build_features(data)

    if scaler is not None:
        try:
            X = scaler.transform(X)
        except Exception as e:
            print(f"[!] Scaler transform failed: {e}")

    if model is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 503

    try:
        probability = float(model.predict_proba(X)[0][1])
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {e}"}), 500

    is_fraud = probability > 0.5
    risk     = "HIGH" if probability > 0.7 else ("MEDIUM" if probability > 0.4 else "LOW")
    amount   = float(data.get("amount", 0))
    email    = data.get("email", "").strip()
    txn_id   = data.get("id", "TXN-UNKNOWN")

    alert_sent = {"email": False, "sms": False}
    if is_fraud and email:
        t = threading.Thread(
            target=send_email_alert,
            args=(email, txn_id, amount, probability, risk),
            daemon=True
        )
        t.start()
        alert_sent["email"] = True

    return jsonify({
        "fraud"      : bool(is_fraud),
        "probability": round(probability, 4),
        "risk"       : risk,
        "alert_sent" : alert_sent
    })

# ─────────────────────────────────────────────
# RUN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 50)
    print("  CardGuard — Fraud Detection API")
    print(f"  http://localhost:{port}")
    print("=" * 50)
    app.run(debug=False, host="0.0.0.0", port=port)
