# ChurnIQ – Flask Backend

## Folder Structure
```
churn-backend/
├── app.py                  ← Flask API server
├── rf_model.pkl            ← Trained Random Forest model
├── one_hot_encoder.pkl     ← OneHotEncoder
├── label_encoder_y.pkl     ← LabelEncoder
├── requirements.txt        ← Python dependencies
└── README.md
```

## Setup & Run

### Step 1 — Install Python dependencies
Open terminal inside this folder and run:
```
pip install -r requirements.txt
```

### Step 2 — Start the Flask server
```
python app.py
```

You should see:
```
🚀 ChurnIQ Flask API starting...
📡 Listening on http://localhost:5000
📊 Model: Random Forest | Features: 45
```

### Step 3 — Test it's working
Open browser and go to: http://localhost:5000
You should see: { "status": "✅ ChurnIQ Random Forest API is running!" }

---

## API Reference

### POST /predict
- **Content-Type:** multipart/form-data
- **Body:** file = <your_csv_file>
- **Response:** JSON array of predictions

### Expected CSV columns:
```
customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
PhoneService, MultipleLines, InternetService, OnlineSecurity,
OnlineBackup, DeviceProtection, TechSupport, StreamingTV,
StreamingMovies, Contract, PaperlessBilling, PaymentMethod,
MonthlyCharges, TotalCharges
```
(Same format as WA_Fn-UseC_-Telco-Customer-Churn.csv)

### Response format:
```json
[
  { "id": 1, "name": "7590-VHVEG", "probability": 82.5, "risk": "High" },
  { "id": 2, "name": "5575-GNVDE", "probability": 45.2, "risk": "Medium" },
  ...
]
```

---

## Troubleshooting

**CORS error** → Make sure Flask server is running before React app  
**Missing column error** → Your CSV must have all 20 columns listed above  
**sklearn version warning** → Safe to ignore, model still works  
