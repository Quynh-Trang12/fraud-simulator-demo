# AnomalyWatchers — DonutPuff 🍩

## Fraud Detection Simulator — COS30049 Assignments 2 & 3

A full-stack, AI-powered fraud detection platform that combines a **tri-model ML pipeline** with a **real-time React dashboard** and **explainable AI (XAI)** feedback.

---

## Architecture

| Layer           | Technology                              | Purpose                                     |
| --------------- | --------------------------------------- | ------------------------------------------- |
| **ML Pipeline** | Python · scikit-learn · XGBoost · SMOTE | Tri-model training with GridSearchCV        |
| **Backend**     | FastAPI · Pydantic · joblib             | Async prediction API with XAI factors       |
| **Frontend**    | React · Vite · TypeScript · Tailwind    | Real-time dashboard & transaction simulator |
| **UI Library**  | shadcn/ui · Recharts · Framer Motion    | Charts, animations, premium components      |

### Tri-Model Architecture

1. **Logistic Regression** (Baseline) — Interpretable linear model with balanced class weights.
2. **XGBoost Classifier** (Champion) — Gradient-boosted trees with `GridSearchCV` hyperparameter tuning; scored using AUPRC (Area Under Precision-Recall Curve).
3. **Isolation Forest** (Unsupervised) — Anomaly detector trained only on legitimate transactions for "unknown unknowns."

---

## Datasets

| Dataset                                         | Source                                                                                     | Role                      | Schema                                                                                     |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------ | ------------------------- | ------------------------------------------------------------------------------------------ |
| **Rupak Roy** — Online Payments Fraud Detection | [Kaggle](https://www.kaggle.com/datasets/rupakroy/online-payments-fraud-detection-dataset) | **Primary (Trainer)**     | `type`, `amount`, `oldbalanceOrg`, `newbalanceOrig`, `errorBalanceOrg`, `errorBalanceDest` |
| **Kartik2112** — Fraud Detection                | [Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)                       | **Secondary (Validator)** | `amt`, `lat`, `long`, `merch_lat`, `merch_long`, `dob`, `city_pop`                         |

### Why Two Datasets?

- **Rupak Roy (Paysim)** simulates mobile money transactions with extreme class imbalance (~0.13% fraud). It defines our core schema and is the primary training source.
- **Kartik2112 (Sparkov)** covers credit card transactions with geospatial features. It validates that the system generalizes across domains and is not overfitting to a single distribution.

---

## Local Setup

### Prerequisites

- **Node.js** ≥ 18 and **npm** ≥ 9
- **Python** ≥ 3.10 with `pip`

### 1. Install Frontend Dependencies

```bash
npm install
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Datasets

Place the CSV files in `data/`:

```
data/
├── onlinefraud.csv      # Rupak Roy (primary)
├── fraudTrain.csv       # Kartik2112 (secondary, training split)
└── fraudTest.csv        # Kartik2112 (secondary, test split)
```

Or run the download helper:

```bash
python scripts/download_data.py
```

### 4. Train ML Models

```bash
# Development mode (10% sample, fast iteration)
python scripts/train_models.py

# Full dataset (production-quality training)
python scripts/train_models.py --full
```

This produces model artifacts in `backend/models/`:

```
backend/models/
├── model_primary.pkl          # XGBoost (Champion)
├── model_logistic.pkl         # Logistic Regression (Baseline)
├── model_isolation_forest.pkl # Isolation Forest (Unsupervised)
└── label_encoder_type.pkl     # Transaction-type encoder
```

### 5. Start the Backend

```bash
uvicorn backend.app.main:app --reload
```

The API starts at `http://localhost:8000`.

### 6. Start the Frontend

```bash
npm run dev
```

The app starts at `http://localhost:8080` (proxies `/predict/*` to the backend).

---

## Key Features

### ML Engineering (Assignment 2)

- **SMOTE** for class imbalance handling (documented justification).
- **GridSearchCV** on XGBoost with `scoring='average_precision'`.
- **AUPRC** (Area Under Precision-Recall Curve) as primary metric — not accuracy.
- Confusion matrix analysis for all three models.

### Full-Stack Architecture (Assignment 3)

- **Async endpoints** (`async def`) for <200ms response times.
- **Startup model loading** via lifespan context manager.
- **Pydantic v2** schemas with `Field()` validators on all inputs.
- **Hybrid detection**: ML probability ensembled with heuristic rules.
- **Structured XAI**: Backend returns typed `RiskFactor` objects for frontend display.

### Frontend

- **Live Dashboard** with Recharts (AreaChart, BarChart, PieChart).
- **Framer Motion** animations on all cards, charts, and modals.
- **Transaction Simulator** with preset scenarios and XAI feedback.
- **Explainability Panel** showing risk factors like "High Amount relative to Old Balance."

---

## Project Structure

```
anomaly-watchers-donutpuff/
├── backend/
│   └── app/
│       ├── main.py              # FastAPI application
│       └── schemas.py           # Pydantic request/response models
├── scripts/
│   ├── train_models.py          # Unified tri-model pipeline
│   └── download_data.py         # Dataset downloader
├── notebooks/
│   ├── 01_primary_analysis.ipynb
│   └── 02_secondary_analysis.ipynb
├── src/                         # React frontend (Vite + TypeScript)
│   ├── components/
│   │   ├── Dashboard.tsx        # Live risk monitor
│   │   └── simulator/           # Transaction form + presets
│   ├── pages/                   # Route pages
│   ├── api.ts                   # Backend API client
│   └── lib/                     # Utilities, scoring, storage
├── requirements.txt             # Python dependencies
├── package.json                 # Node.js dependencies
└── README.md                    # This file
```

---

## Team

**AnomalyWatchers — DonutPuff** · COS30049 · Computing Technology Design Project
