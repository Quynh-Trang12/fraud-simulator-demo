import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import classification_report, average_precision_score, confusion_matrix
from math import radians, cos, sin, asin, sqrt
import joblib
import os

# Config
DATA_PATH = "data/fraudTrain.csv" 
MODEL_DIR = "backend/models"
RF_MODEL_PATH = os.path.join(MODEL_DIR, "model_secondary_rf.pkl")
ISO_MODEL_PATH = os.path.join(MODEL_DIR, "model_secondary_iso.pkl")

def haversine_vectorized(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a)) 
    r = 6371 
    return c * r

def train():
    print("="*60)
    print("SECONDARY MODEL PIPELINE (HD GRADE)")
    print("="*60)
    
    print("\n[Step 1] Loading Secondary Dataset (Sparkov)...")
    if not os.path.exists(DATA_PATH):
        print(f"ERROR: {DATA_PATH} not found.")
        return

    # Sample for speed, but use enough for stability
    df = pd.read_csv(DATA_PATH).sample(n=100000, random_state=42) 
    print(f"Dataset shape: {df.shape}")
    
    print("\n[Step 2] Feature Engineering (Geospatial & Temporal)...")
    # Geospatial: Distance between user and merchant
    df['dist_to_merch'] = haversine_vectorized(df['long'], df['lat'], df['merch_long'], df['merch_lat'])
    
    # Age
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = 2025 - df['dob'].dt.year # Updated to likely current year context

    # Features
    features = ['amt', 'dist_to_merch', 'age', 'city_pop']
    X = df[features]
    y = df['is_fraud']

    # --- MODEL 2: RANDOM FOREST (Champion Challenger) ---
    print("\n[Step 3] Training Random Forest (Supervised Challenger)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Simple tuning for RF
    rf = RandomForestClassifier(class_weight='balanced', random_state=42) # 'balanced' helps with imbalance
    params = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    
    rf_search = RandomizedSearchCV(rf, params, n_iter=3, cv=3, scoring='average_precision', random_state=42)
    rf_search.fit(X_train, y_train)
    best_rf = rf_search.best_estimator_
    
    print("RF Evaluation:")
    probs = best_rf.predict_proba(X_test)[:, 1]
    print(f"RF AUPRC: {average_precision_score(y_test, probs):.4f}")
    print(classification_report(y_test, best_rf.predict(X_test)))
    
    joblib.dump(best_rf, RF_MODEL_PATH)
    print(f"RF Model saved -> {RF_MODEL_PATH}")

    # --- MODEL 3: ISOLATION FOREST (Unsupervised Anomaly Detector) ---
    print("\n[Step 4] Training Isolation Forest (Unsupervised)...")
    print("Justification: We use Isolation Forest to detect 'Unknown Unknowns' - anomalies that deviate from normal behavior but don't match known fraud patterns.")
    
    # Train on X (ignoring labels) to find outliers
    # Contamination = estimate of outlier proportion (e.g. 1%)
    iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)
    iso.fit(X)
    
    # Evaluation (strictly for report, usually IF is hard to evaluate with labels)
    # IF predicts 1 for inlier, -1 for outlier. We map -1 to 1 (Fraud), 1 to 0 (Normal)
    iso_preds = iso.predict(X_test)
    iso_preds_mapped = np.where(iso_preds == -1, 1, 0)
    
    print("Isolation Forest Evaluation (vs Labels):")
    print(confusion_matrix(y_test, iso_preds_mapped))
    print(f"Note: Unsupervised models often have low Precision but high Recall for anomalies.")
    
    joblib.dump(iso, ISO_MODEL_PATH)
    print(f"Isolation Forest Model saved -> {ISO_MODEL_PATH}")

if __name__ == "__main__":
    train()
