from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Allows React frontend at localhost:3000 to connect

# ── Load all pkl files ────────────────────────────────────────────────────────
with open('rf_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model          = model_data['Model']       # RandomForestClassifier
FEATURE_COLS   = model_data['Features']    # exact 45 feature columns in correct order

with open('one_hot_encoder.pkl', 'rb') as f:
    ohe = pickle.load(f)

with open('label_encoder_y.pkl', 'rb') as f:
    le = pickle.load(f)

# ── Categorical columns (same order as training) ──────────────────────────────
OBJECT_COLUMNS = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

# ── Numerical columns ─────────────────────────────────────────────────────────
NUMERIC_COLS = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']


# ── Health check ──────────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "status": "✅ ChurnIQ Random Forest API is running!",
        "endpoint": "POST /predict  →  multipart/form-data  { file: <csv> }"
    })


# ── Predict endpoint ──────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Read uploaded CSV ─────────────────────────────────────────────────
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded. Send CSV as 'file' field."}), 400

        file = request.files['file']
        df   = pd.read_csv(file)

        # 2. Save customer IDs for display ─────────────────────────────────────
        if 'customerID' in df.columns:
            customer_ids = df['customerID'].tolist()
            df = df.drop(columns=['customerID'])
        else:
            customer_ids = [f"Customer {i+1}" for i in range(len(df))]

        # 3. Drop target column if accidentally included ───────────────────────
        if 'Churn' in df.columns:
            df = df.drop(columns=['Churn'])

        # 4. Fix TotalCharges — can be string with spaces in raw data ──────────
        df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

        # 5. One-hot encode categorical columns ────────────────────────────────
        encoded_array = ohe.transform(df[OBJECT_COLUMNS])
        encoded_df    = pd.DataFrame(
            encoded_array,
            columns=ohe.get_feature_names_out(OBJECT_COLUMNS),
            index=df.index
        )

        # 6. Combine numerical + encoded columns ───────────────────────────────
        df_processed = pd.concat(
            [df[NUMERIC_COLS].reset_index(drop=True),
             encoded_df.reset_index(drop=True)],
            axis=1
        )

        # 7. Reorder columns to exactly match training order ───────────────────
        df_processed = df_processed[FEATURE_COLS]

        # 8. Predict ───────────────────────────────────────────────────────────
        probabilities = model.predict_proba(df_processed)[:, 1]  # churn probability

        # 9. Build response ────────────────────────────────────────────────────
        results = []
        for i, prob in enumerate(probabilities):
            prob_percent = round(float(prob) * 100, 1)

            if prob_percent >= 70:
                risk = "High"
            elif prob_percent >= 40:
                risk = "Medium"
            else:
                risk = "Low"

            results.append({
                "id":          i + 1,
                "name":        str(customer_ids[i]),
                "probability": prob_percent,
                "risk":        risk
            })

        # Sort by probability descending
        results = sorted(results, key=lambda x: x['probability'], reverse=True)

        return jsonify(results)

    except KeyError as e:
        return jsonify({"error": f"Missing column in CSV: {str(e)}. Make sure your CSV matches the training data format."}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n🚀 ChurnIQ Flask API starting...")
    print("📡 Listening on http://localhost:5000")
    print("📊 Model: Random Forest | Features:", len(FEATURE_COLS))
    print("─" * 50)
    app.run(debug=True, port=5000)
