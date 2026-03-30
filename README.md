# 🧬 Antibiotic Resistance AI Dashboard

> AI-powered clinical decision support system for antimicrobial resistance prediction and antibiotic recommendation.

---

## 📌 Overview

This project uses machine learning to predict antibiotic resistance in bacteria
and recommend the most effective treatment options based on patient risk factors
and clinical microbiology data.

Built for **CodeCure Hackathon** — fighting the global AMR crisis with AI.

---

## 🚀 Features

- 🎯 **Resistance Prediction** — Predicts Resistant / Intermediate / Susceptible
  with confidence score using XGBoost
- 💊 **Antibiotic Recommendation** — Ranks antibiotics by lowest resistance rate
- 📊 **Drug Comparison** — Full resistance table with visual bars
- 🔍 **Insights Dashboard** — Heatmap, SIR distribution, MDR analysis
- 🧠 **Explainability** — SHAP-style feature contribution analysis
- 📈 **Model Performance** — Accuracy, F1, confusion matrix

---

## 🏗️ Project Structure
```
antibiotic-resistance-ai/
├── app.py                  # Streamlit dashboard (frontend)
├── amr_analysis.py         # Full pipeline runner
├── backend/
│   ├── prediction.py       # predict_resistance()
│   ├── recommendation.py   # recommend_antibiotics()
│   └── main_demo.py        # compare_drugs()
├── src/
│   ├── config.py           # Paths and constants
│   ├── cleaning.py         # Data cleaning
│   ├── features.py         # Feature engineering
│   ├── insights.py         # Insight functions
│   ├── models.py           # Model training
│   ├── visualise.py        # Plot generation
│   ├── drug_comparison.py  # Drug comparison
│   └── explainability.py   # SHAP explanations
├── data/
│   ├── raw/                # Original datasets
│   └── processed/          # Cleaned CSVs
├── outputs/
│   ├── figures/            # Saved plots
│   └── models/             # Saved .pkl models
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/antibiotic-resistance-ai.git
cd antibiotic-resistance-ai
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the full pipeline (train models + generate figures)
```bash
python amr_analysis.py
```

### 4. Launch the dashboard
```bash
python -m streamlit run app.py
```

---

## 📦 Requirements
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
openpyxl
streamlit
jupyter
```

---

## 🤖 Models

| Model | Type | Accuracy |
|-------|------|----------|
| IPM Multi-class (S/I/R) | XGBoost | 66.11% |
| Augmentin Resistance | XGBoost Binary | 77.78% |
| MDR Prediction | XGBoost | 80.28% |

---

## 📊 Dataset

- **Primary dataset** — Clinical microbiology isolates
- **12,400+ isolates** across multiple bacteria species
- **Key organisms** — E. coli, K. pneumoniae, P. aeruginosa, and more
- **Antibiotics tested** — 15 antibiotics across major drug classes

---

## 🧪 Key Findings

- **Most resistant antibiotic** — IPM (69.4% resistance in E. coli)
- **Most effective antibiotic** — Furanes (79.0% susceptible)
- **Most resistant bacteria** — Escherichia coli (38.4% resistance)
- **Top MDR predictor** — bacteria type (87.5% feature importance)

---

## 👥 Team

Built  for **CodeCure Hackathon**

---

## 📄 License

MIT License — free to use and modify.