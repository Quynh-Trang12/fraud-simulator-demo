from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import os
from .schemas import TransactionInput, CreditCardInput, PredictionOutput
from math import radians, cos, sin, asin, sqrt

app = FastAPI(title="Fraud Simulator API", version="1.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8080", "http://localhost:8081"], # Vite default & alternatives
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Models
models = {}

@app.on_event("startup")
def load_models():
    model_dir = "backend/models"
    try:
        models["primary"] = joblib.load(os.path.join(model_dir, "model_primary.pkl"))
        models["encoder"] = joblib.load(os.path.join(model_dir, "label_encoder_type.pkl"))
        models["secondary_rf"] = joblib.load(os.path.join(model_dir, "model_secondary_rf.pkl"))
        # models["secondary_iso"] = joblib.load(os.path.join(model_dir, "model_secondary_iso.pkl"))
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Error loading models: {e}")

@app.get("/")
def health_check():
    return {"status": "ok", "models_loaded": list(models.keys())}

# --- Primary Model (Mobile Money) ---
@app.post("/predict/primary", response_model=PredictionOutput)
def predict_primary(data: TransactionInput):
    if "primary" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Feature Engineering (Must match training script)
    # 1. Type Encoding
    try:
        type_encoded = models["encoder"].transform([data.type])[0]
    except:
        # If unknown type (e.g., DEBIT which we filtered out?), default to CASH_OUT or similar safest
        type_encoded = 0 
    
    # Feature engineering
    errorBalanceOrg = data.newbalanceOrig + data.amount - data.oldbalanceOrg
    errorBalanceDest = data.oldbalanceDest + data.amount - data.newbalanceDest

    features_df = pd.DataFrame([{
        'type': type_encoded,
        'amount': data.amount,
        'oldbalanceOrg': data.oldbalanceOrg,
        'newbalanceOrig': data.newbalanceOrig,
        'errorBalanceOrg': errorBalanceOrg,
        'errorBalanceDest': errorBalanceDest
    }])
    
    # Enforce column order to match model training
    expected_cols = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'errorBalanceOrg', 'errorBalanceDest']
    features = features_df[expected_cols]
    
    print("--- Prediction Request ---")
    print(f"Input: {data}")
    print(f"Features:\n{features}")
    
    # Predict
    prob = models["primary"].predict_proba(features.values)[0][1]
    
    print(f"ML Probability: {prob:.4f}")
    
    # --- Hybrid Detection Logic ---
    # Combine ML score with heuristic rules for specific fraud patterns
    # that might be undersampled in the training data (e.g., forced overdrafts).
    
    heuristic_prob = 0.0
    reasons = []
    
    # Rule 1: Forced Overdraft (Negative Balance)
    # This specifically catches the "High Risk" preset where Amount > Balance
    if data.newbalanceOrig < 0:
         heuristic_prob = max(heuristic_prob, 0.99)
         reasons.append("Illegal Overdraft (Negative Balance) detected")

    # Rule 2: Zero-out / Balance Drain
    # If balance hits exactly 0, check if it was a total drain
    elif data.newbalanceOrig == 0 and data.amount > 0:
        if data.amount >= data.oldbalanceOrg:
             heuristic_prob = max(heuristic_prob, 0.95)
             reasons.append("Balance drain pattern detected")

    # Rule 3: Balance Error (Mathematical anomaly = Potential Exploit)
    # Standard check: new + amount - old should be 0 (for drain) or diff
    # But for Overdraft, the math IS valid (50k - 99k = -49k), so Rule 1 catches it.
    if abs(errorBalanceOrg) > 0.01:
        # If the math doesn't add up (e.g. overdrafting more than balance), flag it
        heuristic_prob = max(heuristic_prob, 0.85)
        reasons.append(f"Balance discrepancy ({errorBalanceOrg:.2f})")

    # Rule 4: High Value
    if data.amount > 150000:
        heuristic_prob = max(heuristic_prob, 0.70)
        reasons.append("High value transaction")

    # Final Score: Ensemble (Max of ML and Rules)
    final_prob = max(prob, heuristic_prob)
    
    # --- Risk Assessment ---
    is_fraud = final_prob > 0.5
    
    risk_level = "Low"
    if final_prob > 0.8:
        risk_level = "High"
    elif final_prob > 0.4:
        risk_level = "Medium"
    
    # Generate explanations
    # Use the ML reasons if ML score is high, otherwise use Rule reasons
    ml_reasons = []
    if prob > 0.5:
        ml_reasons.append("AI Model detected suspicious pattern")
    
    all_reasons = list(set(reasons + ml_reasons))
    
    explanation_text = "Transaction appears safe." if not is_fraud else f"Risk Factors: {', '.join(all_reasons)}."

    return {
        "probability": float(prob),
        "is_fraud": bool(is_fraud),
        "risk_level": risk_level,
        "explanation": explanation_text
    }

# --- Secondary Model (Credit Card) ---
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 
    return c * r

@app.post("/predict/secondary", response_model=PredictionOutput)
def predict_secondary(data: CreditCardInput):
    if "secondary_rf" not in models:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # Feature Engineering
    dist = haversine(data.long, data.lat, data.merch_long, data.merch_lat)
    
    try:
        data_dob = pd.to_datetime(data.dob)
        age = 2020 - data_dob.year
    except:
        age = 30 # default
        
    features = pd.DataFrame([{
        'amt': data.amt,
        'dist_to_merch': dist,
        'age': age,
        'city_pop': data.city_pop
    }])
    
    # Predict
    pred = models["secondary_rf"].predict(features)[0]
    prob = models["secondary_rf"].predict_proba(features)[0][1]
    
    return {
        "probability": float(prob),
        "is_fraud": bool(pred),
        "risk_level": "High" if prob > 0.7 else "Medium" if prob > 0.3 else "Low",
        "explanation": f"Distance: {dist:.2f}km"
    }
