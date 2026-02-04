import joblib
import os
import sys
import pandas as pd

# Fix path to find models
if not os.path.exists("backend/models/model_primary.pkl"):
    print("Cannot find model file at backend/models/model_primary.pkl")
    sys.exit(1)

try:
    model = joblib.load("backend/models/model_primary.pkl")
    encoder = joblib.load("backend/models/label_encoder_type.pkl")
    print("--- Models Loaded ---")

    # Simulate Input
    # Step 1: Encode Type
    input_type = "PAYMENT"
    try:
        type_encoded = encoder.transform([input_type])[0]
        print(f"Encoded type '{input_type}' -> {type_encoded}")
    except Exception as e:
        print(f"Encoder Error: {e}")
        type_encoded = 0

    with open("debug_output.txt", "w") as f:
        f.write(f"Encoder Classes: {list(encoder.classes_)}\n")

        scenarios = [
            {
                "name": "User Scenario (Overdraft)",
                "data": {
                    'type': "TRANSFER",
                    'amount': 99000.0,
                    'oldbalanceOrg': 50000.0,
                    'newbalanceOrig': 0.0,
                    'oldbalanceDest': 0.0, 
                    'newbalanceDest': 0.0 # Dest details usually less relevant
                }
            },
            {
                "name": "Exact Drain (Valid Math)",
                "data": {
                    'type': "TRANSFER",
                    'amount': 50000.0,
                    'oldbalanceOrg': 50000.0,
                    'newbalanceOrig': 0.0,
                    'oldbalanceDest': 0.0, 
                    'newbalanceDest': 0.0
                }
            },
            {
                "name": "Typical Fraud (Pattern)",
                "data": {
                    'type': "TRANSFER",
                    'amount': 50000.0,
                    'oldbalanceOrg': 50000.0,
                    'newbalanceOrig': 50000.0, # NO CHANGE (Theft often leaves balance high in some datasets?) Or new=0?
                    'oldbalanceDest': 0.0, 
                    'newbalanceDest': 0.0
                }
            }
        ]

        f.write("\n--- Scenario Testing ---\n")
        
        for sc in scenarios:
            f.write(f"\nScenario: {sc['name']}\n")
            d = sc['data']
            
            # Encode type
            try:
                t_enc = encoder.transform([d['type']])[0]
            except:
                t_enc = 0
            
            # Features
            errorOrg = d['newbalanceOrig'] + d['amount'] - d['oldbalanceOrg']
            errorDest = d['oldbalanceDest'] + d['amount'] - d['newbalanceDest']
            
            vec = pd.DataFrame([{
                'type': int(t_enc),
                'amount': d['amount'],
                'oldbalanceOrg': d['oldbalanceOrg'],
                'newbalanceOrig': d['newbalanceOrig'],
                'errorBalanceOrg': errorOrg,
                'errorBalanceDest': errorDest
            }])
            
            # Enforce Order
            cols = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'errorBalanceOrg', 'errorBalanceDest']
            vec = vec[cols]
            
            f.write(f"Features: {vec.values.tolist()}\n")
            f.write(f"ErrorOrg: {errorOrg}, ErrorDest: {errorDest}\n")
            
            prob = model.predict_proba(vec.values)[0][1]
            f.write(f"Risk Score: {prob * 100:.2f}%\n")

except Exception as e:
    with open("debug_output.txt", "a") as f:
        f.write(f"\nOuter Error: {e}\n")
