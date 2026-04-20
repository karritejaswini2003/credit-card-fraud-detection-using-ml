"""
FraudShield ML — Credit Card Fraud Detection System
====================================================
Run:
    pip install -r requirements.txt
    python app.py
Open: http://localhost:5000
Credentials:
    Admin  → username: admin   | password: admin123
    User   → username: ravi    | password: ravi123
"""

from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import random, time, json, os
from datetime import datetime, timedelta
from functools import wraps
import numpy as np

app = Flask(__name__)
app.secret_key = 'fraudshield-ml-secret-2025'
CORS(app, supports_credentials=True)

# ══════════════════════════════════════════════════════════════
#  IN-MEMORY STORE  (swap with SQLite/Postgres in production)
# ══════════════════════════════════════════════════════════════

USERS = {
    "admin": {"password": "admin123", "role": "admin",    "name": "Arjun Reddy",  "email": "admin@fraudshield.com",  "phone": "+91 9876543210"},
    "ravi":  {"password": "ravi123",  "role": "customer", "name": "Ravi Kumar",   "email": "ravi@example.com",       "phone": "+91 8765432109"},
    "priya": {"password": "priya123", "role": "customer", "name": "Priya Sharma", "email": "priya@example.com",      "phone": "+91 7654321098"},
}

CUSTOMERS = {
    "C001": {"name": "Ravi Kumar",    "email": "ravi@example.com",    "phone": "+91 9876543210", "card": "**** **** **** 4521", "status": "active",  "risk": "LOW",    "balance": 42500.00, "joined": "2023-01-15"},
    "C002": {"name": "Priya Sharma",  "email": "priya@example.com",   "phone": "+91 8765432109", "card": "**** **** **** 7832", "status": "active",  "risk": "MEDIUM", "balance": 18900.00, "joined": "2023-03-22"},
    "C003": {"name": "Arjun Mehta",   "email": "arjun@example.com",   "phone": "+91 7654321098", "card": "**** **** **** 2341", "status": "blocked", "risk": "HIGH",   "balance": 5200.00,  "joined": "2022-11-05"},
    "C004": {"name": "Meera Nair",    "email": "meera@example.com",   "phone": "+91 6543210987", "card": "**** **** **** 9087", "status": "active",  "risk": "LOW",    "balance": 67800.00, "joined": "2023-06-10"},
    "C005": {"name": "Suresh Babu",   "email": "suresh@example.com",  "phone": "+91 5432109876", "card": "**** **** **** 6543", "status": "active",  "risk": "MEDIUM", "balance": 31200.00, "joined": "2022-08-30"},
    "C006": {"name": "Divya Menon",   "email": "divya@example.com",   "phone": "+91 4321098765", "card": "**** **** **** 1122", "status": "active",  "risk": "LOW",    "balance": 55000.00, "joined": "2023-09-14"},
    "C007": {"name": "Kiran Rao",     "email": "kiran@example.com",   "phone": "+91 3210987654", "card": "**** **** **** 3344", "status": "active",  "risk": "HIGH",   "balance": 8700.00,  "joined": "2024-01-20"},
}

TRANSACTIONS = []
ALERTS = []
BLOCKED_CARDS = [
    {"card": "**** **** **** 2341", "customer": "Arjun Mehta",  "reason": "Multiple fraud attempts", "date": "2025-01-10", "amount": "₹18,450", "cid": "C003"},
    {"card": "**** **** **** 8812", "customer": "Unknown",       "reason": "Stolen card reported",    "date": "2025-01-15", "amount": "₹2,100",  "cid": None},
]

# Seed realistic transactions
def seed_transactions():
    samples = [
        {"id":"TXN-100001","customer":"C001","name":"Ravi Kumar",   "amount":1499.00,"fraud":False,"prob":0.04,"risk":"LOW",   "time":"09:10:05","location":"Hyderabad","merchant":"Amazon"},
        {"id":"TXN-100002","customer":"C002","name":"Priya Sharma", "amount":2.69,   "fraud":True, "prob":0.91,"risk":"HIGH",  "time":"09:23:14","location":"Unknown",   "merchant":"Unknown"},
        {"id":"TXN-100003","customer":"C004","name":"Meera Nair",   "amount":3120.00,"fraud":False,"prob":0.06,"risk":"LOW",   "time":"09:35:33","location":"Bengaluru", "merchant":"Flipkart"},
        {"id":"TXN-100004","customer":"C005","name":"Suresh Babu",  "amount":449.99, "fraud":False,"prob":0.12,"risk":"LOW",   "time":"10:02:48","location":"Chennai",   "merchant":"Swiggy"},
        {"id":"TXN-100005","customer":"C003","name":"Arjun Mehta",  "amount":3.00,   "fraud":True, "prob":0.95,"risk":"HIGH",  "time":"10:15:02","location":"Unknown",   "merchant":"Unknown"},
        {"id":"TXN-100006","customer":"C001","name":"Ravi Kumar",   "amount":880.00, "fraud":False,"prob":0.07,"risk":"LOW",   "time":"10:31:19","location":"Hyderabad", "merchant":"BigBasket"},
        {"id":"TXN-100007","customer":"C002","name":"Priya Sharma", "amount":2150.00,"fraud":False,"prob":0.19,"risk":"LOW",   "time":"10:44:45","location":"Pune",      "merchant":"Myntra"},
        {"id":"TXN-100008","customer":"C005","name":"Suresh Babu",  "amount":1.50,   "fraud":True, "prob":0.88,"risk":"HIGH",  "time":"11:05:11","location":"Foreign",   "merchant":"Unknown"},
        {"id":"TXN-100009","customer":"C006","name":"Divya Menon",  "amount":5600.00,"fraud":False,"prob":0.03,"risk":"LOW",   "time":"11:20:30","location":"Mumbai",    "merchant":"Apple Store"},
        {"id":"TXN-100010","customer":"C007","name":"Kiran Rao",    "amount":4.50,   "fraud":True, "prob":0.79,"risk":"HIGH",  "time":"11:33:55","location":"Unknown",   "merchant":"Unknown"},
    ]
    for s in samples:
        TRANSACTIONS.append(s)
        if s["fraud"]:
            ALERTS.append({
                "id": "ALT-" + str(random.randint(1000, 9999)),
                "txn": s["id"], "customer": s["name"], "amount": s["amount"],
                "risk": s["risk"], "time": s["time"], "status": "sent",
                "email": CUSTOMERS.get(s["customer"], {}).get("email", ""),
                "phone": CUSTOMERS.get(s["customer"], {}).get("phone", ""),
                "read": False
            })

seed_transactions()

# ══════════════════════════════════════════════════════════════
#  ML SIMULATION (replace with real model)
# ══════════════════════════════════════════════════════════════

def predict_fraud(v1, v4, v12, v14, v17, amount):
    """
    Simulated XGBoost prediction.
    To use a real model:
        import joblib
        model = joblib.load('models/fraud_model.pkl')
        prob  = model.predict_proba([[v1,v4,v12,v14,v17,amount]])[0][1]
    """
    p = 0.04
    if abs(v14) > 5:    p += 0.38
    if abs(v14) > 10:   p += 0.26
    if v4 < -2:         p += 0.20
    if abs(v12) > 5:    p += 0.15
    if abs(v17) > 3:    p += 0.10
    if 0 < amount < 5:  p += 0.08
    if v1 < -3:         p += 0.12
    p += random.uniform(-0.02, 0.02)
    return round(max(0.02, min(0.98, p)), 4)

# Feature importance (for display)
FEATURE_IMPORTANCE = {
    "V14 (PCA Feature 14)":  38.2,
    "V12 (PCA Feature 12)":  19.7,
    "V4  (PCA Feature 4)":   15.3,
    "V17 (PCA Feature 17)":  12.1,
    "V1  (PCA Feature 1)":    8.9,
    "Amount":                  5.8,
}

# ══════════════════════════════════════════════════════════════
#  AUTH DECORATORS
# ══════════════════════════════════════════════════════════════

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session or session.get('role') != 'admin':
            return jsonify({"error": "Admin only"}), 403
        return f(*args, **kwargs)
    return decorated

# ══════════════════════════════════════════════════════════════
#  ROUTES — PAGES
# ══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "online", "version": "3.0", "timestamp": time.time()})

# ══════════════════════════════════════════════════════════════
#  ROUTES — AUTH
# ══════════════════════════════════════════════════════════════

@app.route('/api/login', methods=['POST'])
def login():
    data     = request.get_json(force=True)
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()
    user     = USERS.get(username)
    if not user or user['password'] != password:
        return jsonify({"success": False, "message": "Invalid username or password"}), 401
    session['user']  = username
    session['role']  = user['role']
    session['name']  = user['name']
    session['email'] = user['email']
    session['phone'] = user['phone']
    return jsonify({"success": True, "role": user['role'], "name": user['name'],
                    "email": user['email'], "phone": user['phone']})

@app.route('/api/logout', methods=['POST'])
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route('/api/me')
def me():
    if 'user' not in session:
        return jsonify({"logged_in": False})
    return jsonify({"logged_in": True, "username": session['user'],
                    "role": session['role'], "name": session['name'],
                    "email": session.get('email',''), "phone": session.get('phone','')})

# ══════════════════════════════════════════════════════════════
#  ROUTES — PREDICT
# ══════════════════════════════════════════════════════════════

@app.route('/api/predict', methods=['POST'])
def predict():
    data   = request.get_json(force=True)
    amount = float(data.get('amount', 0))
    v1     = float(data.get('v1', 0))
    v4     = float(data.get('v4', 0))
    v12    = float(data.get('v12', 0))
    v14    = float(data.get('v14', 0))
    v17    = float(data.get('v17', 0))
    email  = data.get('email', '').strip()
    phone  = data.get('phone', '').strip()
    cid    = data.get('customer_id', 'C001')
    name   = data.get('name', '').strip()
    txn_id = 'TXN-' + str(random.randint(100000, 999999))
    now    = datetime.now().strftime('%H:%M:%S')

    prob     = predict_fraud(v1, v4, v12, v14, v17, amount)
    is_fraud = prob > 0.5
    risk     = 'HIGH' if prob > 0.7 else ('MEDIUM' if prob > 0.4 else 'LOW')

    cdata = CUSTOMERS.get(cid, {})
    cname = name or cdata.get('name', 'Unknown')

    txn = {
        "id": txn_id, "customer": cid, "name": cname,
        "amount": amount, "fraud": is_fraud, "prob": prob,
        "risk": risk, "time": now,
        "location": data.get('location', 'Unknown'),
        "merchant": data.get('merchant', 'Unknown')
    }
    TRANSACTIONS.insert(0, txn)
    if len(TRANSACTIONS) > 500:
        TRANSACTIONS.pop()

    alert_sent = {"email": False, "sms": False}
    if is_fraud:
        alert = {
            "id": "ALT-" + str(random.randint(1000, 9999)),
            "txn": txn_id, "customer": cname, "amount": amount,
            "risk": risk, "time": now, "status": "sent",
            "email": email, "phone": phone, "read": False
        }
        ALERTS.insert(0, alert)
        alert_sent = {"email": bool(email), "sms": bool(phone)}
        if email: print(f"[ALERT EMAIL] → {email} | {txn_id} | ₹{amount:.2f} | {prob*100:.1f}%")
        if phone: print(f"[ALERT SMS]   → {phone} | {txn_id} | Risk: {risk}")

    return jsonify({
        "id": txn_id, "fraud": is_fraud, "probability": prob,
        "risk": risk, "amount": amount, "alert_sent": alert_sent,
        "features": {"v1": v1, "v4": v4, "v12": v12, "v14": v14, "v17": v17}
    })

# ══════════════════════════════════════════════════════════════
#  ROUTES — CUSTOMERS
# ══════════════════════════════════════════════════════════════

@app.route('/api/customers')
def get_customers():
    return jsonify([{"id": k, **v} for k, v in CUSTOMERS.items()])

@app.route('/api/customers/<cid>')
def get_customer(cid):
    c = CUSTOMERS.get(cid)
    if not c:
        return jsonify({"error": "Not found"}), 404
    cust_txns = [t for t in TRANSACTIONS if t.get('customer') == cid]
    return jsonify({"id": cid, **c, "transaction_count": len(cust_txns)})

@app.route('/api/customers/add', methods=['POST'])
def add_customer():
    data = request.get_json(force=True)
    cid  = 'C' + str(random.randint(100, 999))
    while cid in CUSTOMERS:
        cid = 'C' + str(random.randint(100, 999))
    CUSTOMERS[cid] = {
        "name":    data.get('name', 'Unknown'),
        "email":   data.get('email', ''),
        "phone":   data.get('phone', ''),
        "card":    "**** **** **** " + str(random.randint(1000, 9999)),
        "status":  "active",
        "risk":    "LOW",
        "balance": float(data.get('balance', 0)),
        "joined":  datetime.now().strftime('%Y-%m-%d')
    }
    return jsonify({"success": True, "id": cid, **CUSTOMERS[cid]})

@app.route('/api/customers/<cid>/block', methods=['POST'])
def block_customer(cid):
    if cid in CUSTOMERS:
        CUSTOMERS[cid]['status'] = 'blocked'
        CUSTOMERS[cid]['risk']   = 'HIGH'
        BLOCKED_CARDS.insert(0, {
            "card":     CUSTOMERS[cid]['card'],
            "customer": CUSTOMERS[cid]['name'],
            "reason":   "Manually blocked by admin",
            "date":     datetime.now().strftime('%Y-%m-%d'),
            "amount":   "—", "cid": cid
        })
    return jsonify({"success": True})

@app.route('/api/customers/<cid>/unblock', methods=['POST'])
def unblock_customer(cid):
    if cid in CUSTOMERS:
        CUSTOMERS[cid]['status'] = 'active'
        CUSTOMERS[cid]['risk']   = 'LOW'
        BLOCKED_CARDS[:] = [b for b in BLOCKED_CARDS if b.get('cid') != cid]
    return jsonify({"success": True})

# ══════════════════════════════════════════════════════════════
#  ROUTES — TRANSACTIONS / ALERTS / BLOCKED
# ══════════════════════════════════════════════════════════════

@app.route('/api/transactions')
def get_transactions():
    return jsonify(TRANSACTIONS[:100])

@app.route('/api/alerts')
def get_alerts():
    return jsonify(ALERTS[:50])

@app.route('/api/alerts/<aid>/read', methods=['POST'])
def mark_alert_read(aid):
    for a in ALERTS:
        if a['id'] == aid:
            a['read'] = True
    return jsonify({"success": True})

@app.route('/api/blocked')
def get_blocked():
    return jsonify(BLOCKED_CARDS)

# ══════════════════════════════════════════════════════════════
#  ROUTES — ANALYTICS
# ══════════════════════════════════════════════════════════════

@app.route('/api/analytics')
def analytics():
    total  = len(TRANSACTIONS)
    frauds = [t for t in TRANSACTIONS if t.get('fraud')]
    safes  = [t for t in TRANSACTIONS if not t.get('fraud')]
    total_amount = sum(t['amount'] for t in TRANSACTIONS)
    fraud_amount = sum(t['amount'] for t in frauds)
    avg_prob     = round(sum(t['prob'] for t in TRANSACTIONS) / max(total, 1), 3)
    risk_counts  = {"HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for t in TRANSACTIONS:
        risk_counts[t.get('risk', 'LOW')] += 1
    daily = {}
    for i in range(7):
        d = (datetime.now() - timedelta(days=i)).strftime('%d %b')
        daily[d] = {"fraud": random.randint(1, 5), "safe": random.randint(8, 25)}
    return jsonify({
        "total": total, "fraud_count": len(frauds), "safe_count": len(safes),
        "total_amount": round(total_amount, 2), "fraud_amount": round(fraud_amount, 2),
        "fraud_rate": round(len(frauds) / max(total, 1) * 100, 1),
        "avg_probability": avg_prob, "risk_counts": risk_counts, "daily": daily,
        "blocked_cards": len(BLOCKED_CARDS), "alerts_sent": len(ALERTS),
        "feature_importance": FEATURE_IMPORTANCE,
        "accuracy": 97.8, "precision": 96.2, "recall": 94.5, "f1": 95.3
    })

# ══════════════════════════════════════════════════════════════
#  ROUTES — EXPORT / SAMPLES
# ══════════════════════════════════════════════════════════════

@app.route('/api/export/transactions')
def export_transactions():
    from flask import Response
    rows = ["ID,Customer,Amount,Fraud,Risk,Probability,Time,Location,Merchant"]
    for t in TRANSACTIONS:
        rows.append(f"{t['id']},{t['name']},{t['amount']},{t['fraud']},{t['risk']},{t['prob']},{t['time']},{t.get('location','')},{t.get('merchant','')}")
    csv = "\n".join(rows)
    return Response(csv, mimetype='text/csv',
                    headers={"Content-Disposition": "attachment;filename=fraud_transactions.csv"})

@app.route('/api/samples')
def samples():
    return jsonify({
        "fraud": {"amount": 2.69,  "v1": -3.04, "v4": -4.28, "v12": -9.50,  "v14": -16.89, "v17": -3.75, "location": "Foreign",    "merchant": "Unknown"},
        "legit": {"amount": 149.62,"v1": -1.36, "v4":  1.38, "v12": -0.185, "v14": -0.311,  "v17":  0.207,"location": "Hyderabad",  "merchant": "Amazon"}
    })

# ══════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "="*55)
    print("  FraudShield ML — v3.0 Starting...")
    print("  URL   : http://localhost:5000")
    print("  Admin : admin / admin123")
    print("  User  : ravi  / ravi123  |  priya / priya123")
    print("="*55 + "\n")
    app.run(debug=True, port=5000, host='0.0.0.0')
