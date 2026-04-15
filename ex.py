# ======================
# 1. Import Libraries
# ======================
import pandas as pd
import numpy as np
import os

os.chdir(os.path.dirname(__file__))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, roc_curve
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# 2. Load & Preprocess
# ======================
df = pd.read_csv("kag_risk_factors_cervical_cancer.csv")
df = df.replace("?", np.nan)
df = df.apply(pd.to_numeric)
df = df.fillna(df.median())

# ======================
# 3. Target & Features
# ======================
y = df["Biopsy"]
X = df.drop(columns=["Biopsy", "Hinselmann", "Schiller", "Citology"], errors='ignore')
print("Class distribution:\n", y.value_counts())

# ======================
# 4. Train-Test Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 5. Standardization
# ======================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# 6. Define Models (NO SMOTE, NO threshold tuning)
# ======================
models = {
    "SVC": SVC(probability=True),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(random_state=42),
    "XGBoost": XGBClassifier(eval_metric='logloss', random_state=42)
}

# ======================
# 7. Train & Evaluate
# ======================
results = {}
roc_data = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)          # Default 0.5 threshold
    y_prob = model.predict_proba(X_test)[:, 1]

    results[name] = [
        accuracy_score(y_test, y_pred),
        recall_score(y_test, y_pred),
        f1_score(y_test, y_pred),
        roc_auc_score(y_test, y_prob)
    ]

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr)

# ======================
# 8. Results Table
# ======================
results_df = pd.DataFrame(results, index=["Accuracy", "Recall", "F1", "AUC"]).T
print("\n===== BASELINE Model Performance =====")
print(results_df.round(4))

# Save for later comparison
results_df.to_csv("baseline_results.csv")