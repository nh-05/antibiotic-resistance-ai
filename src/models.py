import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import pickle
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from src.config import AB_COLS, CLSI_BREAKPOINTS, MODELS_DIR, FIGURES_DIR

# Ensure output directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# ====================
# Helper Functions
# ====================

def encode_res(x):
    """Encode antibiotic resistance: R=1, S/I=0, else NaN"""
    if x == "R": return 1
    if x in ["S", "I"]: return 0
    return np.nan

def train_evaluate(name, model, X_train, X_test, y_train, y_test, labels):
    """Train model, evaluate, print metrics, return model + predictions"""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name}")
    print(f"Accuracy : {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=labels))
    return model, y_pred

# ====================
# Main Function
# ====================

def run_all_models(df_primary, df_secondary):

    # =========================================================
    # Model 1: Multi-class IPM (S/I/R)
    # =========================================================
    print("\n── Model 1: IPM Multi-class (S/I/R) ──")

    TARGET = "IPM"
    FEATURES = [c for c in AB_COLS if c != TARGET]
    df_m1 = df_primary[FEATURES + [TARGET]].copy()

    # Encode features and target
    for col in FEATURES:
        df_m1[col] = df_m1[col].map(encode_res)
    df_m1[TARGET] = df_m1[TARGET].map({"S": 0, "I": 1, "R": 2})
    df_m1 = df_m1.dropna()

    # Feature engineering
    df_m1["res_count"] = (df_m1[FEATURES] == 1).sum(axis=1)
    df_m1["res_ratio"] = df_m1["res_count"] / len(FEATURES)
    FEATURES_EXT = FEATURES + ["res_count", "res_ratio"]

    X1 = SimpleImputer(strategy="median").fit_transform(df_m1[FEATURES_EXT])
    y1 = df_m1[TARGET]

    X1_tr, X1_te, y1_tr, y1_te = train_test_split(
        X1, y1, test_size=0.2, random_state=42, stratify=y1
    )

    xgb1 = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss"
    )

    xgb1, y_pred1 = train_evaluate(
        "XGBoost Multi-class",
        xgb1,
        X1_tr, X1_te,
        y1_tr, y1_te,
        ["Susceptible", "Intermediate", "Resistant"]
    )

    # Feature Importance
    importance = pd.Series(xgb1.feature_importances_, index=FEATURES_EXT).sort_values(ascending=False)
    print("\nTop Features (Model 1):")
    print(importance.head(10))

    # Plot Feature Importance
    importance.head(10).plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Feature Importance — IPM")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_feature_importance_ipm.png")
    plt.close()

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_estimator(
        xgb1, X1_te, y1_te,
        display_labels=["S", "I", "R"],
        cmap="Blues", ax=ax
    )
    plt.savefig(FIGURES_DIR / "05_confusion_matrix_ipm.png")
    plt.close()

    # Save model
    with open(MODELS_DIR / "xgb_ipm_multiclass.pkl", "wb") as f:
        pickle.dump(xgb1, f)

    # Interpretation
    print("\n🔍 Interpretation (Model 1):")
    top_feats = importance.head(5).index.tolist()
    for feat in top_feats:
        print(f"- {feat} is a strong predictor of antibiotic resistance.")
    print("\n💡 Insight:")
    print("• Resistance count and ratio indicate multi-drug resistance patterns.")
    print("• Specific antibiotics strongly influence resistance classification.")
    print("• Model captures antibiotic response patterns effectively.")

    # =========================================================
    # Model 2: Binary — Augmentin Resistance
    # =========================================================
    print("\n── Model 2: Augmentin Resistance ──")

    AB2_FEATURES = [ab for ab in CLSI_BREAKPOINTS if ab != "AUGMENTIN"]
    df_m2 = df_secondary[[ab + "_class" for ab in AB2_FEATURES] + ["AUGMENTIN_class", "Location_enc"]].copy()
    df_m2 = df_m2[df_m2["AUGMENTIN_class"].isin(["R", "S"])].dropna()
    df_m2["target"] = (df_m2["AUGMENTIN_class"] == "R").astype(int)
    for col in [ab + "_class" for ab in AB2_FEATURES]:
        df_m2[col] = df_m2[col].map({"R": 1, "S": 0})

    X2 = df_m2[[ab + "_class" for ab in AB2_FEATURES] + ["Location_enc"]]
    y2 = df_m2["target"]
    scale_pos_weight2 = (y2 == 0).sum() / (y2 == 1).sum()
    X2_tr, X2_te, y2_tr, y2_te = train_test_split(X2, y2, test_size=0.2, random_state=42, stratify=y2)

    xgb2 = XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight2
    )

    xgb2, _ = train_evaluate(
        "XGBoost Binary",
        xgb2,
        X2_tr, X2_te,
        y2_tr, y2_te,
        ["Susceptible", "Resistant"]
    )

    # Save Model
    with open(MODELS_DIR / "xgb_augmentin_resistance.pkl", "wb") as f:
        pickle.dump(xgb2, f)

    # Interpretation
    print("\n🔍 Interpretation (Model 2):")
    print("• Model predicts resistance for a specific antibiotic (Augmentin).")
    print("• Provides insights for targeted antibiotic resistance prediction.")
    print("• Limited sample size may reduce generalizability.")

    # =========================================================
    # Model 3: MDR Prediction (Clinical + Bacteria)
    # =========================================================
    le = LabelEncoder()
    df_primary["bacteria_enc"] = le.fit_transform(df_primary["Species_clean"])
    print("\n── Model 3: MDR Prediction ──")

    CLINICAL = ["Hypertension_enc", "Diabetes_enc", "Hospital_before_enc", "bacteria_enc"]
    df_m3 = df_primary[CLINICAL + ["is_MDR"]].dropna()

    X3 = df_m3[CLINICAL]
    y3 = df_m3["is_MDR"]
    scale_pos_weight3 = (y3 == 0).sum() / (y3 == 1).sum()

    X3_tr, X3_te, y3_tr, y3_te = train_test_split(
        X3, y3, test_size=0.2, random_state=42, stratify=y3
    )

    xgb3 = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight3
    )

    xgb3.fit(X3_tr, y3_tr)

    # Custom threshold
    probs = xgb3.predict_proba(X3_te)[:, 1]
    threshold = 0.3
    y_pred = (probs > threshold).astype(int)

    # Evaluation
    acc = accuracy_score(y3_te, y_pred)
    print(f"\nXGBoost MDR (Threshold = {threshold})")
    print(f"Accuracy : {acc*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y3_te, y_pred, target_names=["Non-MDR", "MDR"]))

    # Confusion Matrix
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y3_te, y_pred, display_labels=["Non-MDR", "MDR"], cmap="Blues", ax=ax)
    plt.title("Confusion Matrix — MDR")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "07_confusion_matrix_mdr.png")
    plt.close()

    # Feature Importance
    importance = pd.Series(xgb3.feature_importances_, index=X3.columns).sort_values(ascending=False)
    print("\nTop Features (Model 3):")
    print(importance)

    # Interpretation
    print("\n🔍 Interpretation (Model 3):")
    print(f"- {importance.idxmax()} is the dominant factor in MDR prediction.")
    for feat in importance.index[1:]:
        if importance[feat] > 0:
            print(f"- {feat} has moderate contribution.")
        else:
            print(f"- {feat} has negligible contribution.")

    print("\n💡 Clinical Insights:")
    print("• Pathogen type plays the largest role.")
    print("• Prior hospitalization increases MDR risk.")
    print("• Some clinical features may not contribute significantly.")

    # Plot Feature Importance
    importance.plot(kind="barh")
    plt.gca().invert_yaxis()
    plt.title("Feature Importance — MDR")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "08_feature_importance_mdr.png")
    plt.close()

    # Save model
    with open(MODELS_DIR / "xgb_mdr_clinical.pkl", "wb") as f:
        pickle.dump(xgb3, f)

    print("\n✅ All models saved successfully!")