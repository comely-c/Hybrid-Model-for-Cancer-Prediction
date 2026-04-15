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

from imblearn.combine import SMOTEENN

import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# 2. Load Dataset
# ======================
df = pd.read_csv("kag_risk_factors_cervical_cancer.csv")

# ======================
# 3. Preprocessing
# ======================
df = df.replace("?", np.nan)
df = df.apply(pd.to_numeric)
df = df.fillna(df.median())

# ======================
# 4. Target & Features
# ======================
y = df["Biopsy"]
X = df.drop(columns=["Biopsy", "Hinselmann", "Schiller", "Citology"], errors='ignore')

print(y.value_counts())

# ======================
# 5. Train-Test Split
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ======================
# 6. Standardization
# ======================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ======================
# 7. SMOTE
# ======================
smote = SMOTEENN(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ======================
# 8. Models
# ======================
models = {
    "SVC": SVC(probability=True,class_weight='balanced'),
    "KNN": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(class_weight='balanced'),
    "XGBoost": XGBClassifier(eval_metric='logloss')
}

# ======================
# 9. Training & Evaluation
# ======================
results = {}
roc_data = {}

for name, model in models.items():
    model.fit(X_train, y_train)

    # y_pred = model.predict(X_test)
    # y_prob = model.predict_proba(X_test)[:, 1]
    y_prob = model.predict_proba(X_test)[:,1]
    y_pred = (y_prob > 0.3).astype(int)

    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    results[name] = [acc, recall, f1, auc]

    # ROC data
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr)

# ======================
# 10. Results Table
# ======================
results_df = pd.DataFrame(results, index=["Accuracy", "Recall", "F1", "AUC"]).T
print("\n===== Model Performance =====")
print(results_df)

# ======================
# 11. Bar Plot (Model Comparison)
# ======================
results_df.plot(kind='bar', figsize=(10,6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("model_comparison.png")
plt.show()

# ======================
# 12. ROC Curve
# ======================
plt.figure(figsize=(8,6))

for name, (fpr, tpr) in roc_data.items():
    plt.plot(fpr, tpr, label=name)

plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()

# ======================
# 13. Confusion Matrix (Best Model = XGBoost or best AUC)
# ======================
best_model_name = results_df["AUC"].idxmax()
best_model = models[best_model_name]

y_pred_best = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.show()