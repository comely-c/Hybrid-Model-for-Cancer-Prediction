# ============================================================
# Cervical Cancer Early Prediction - Full Pipeline
# ============================================================

import pandas as pd
import numpy as np
import os
import time
import warnings
warnings.filterwarnings("ignore")

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Core ML
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)

# Models
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier

# Imbalance
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE   # <-- 新增，用于 baseline（参考代码做法）

# Visualization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# SHAP (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ModuleNotFoundError:
    SHAP_AVAILABLE = False
    print("Warning: shap module not found. Skipping SHAP analysis.")


# ============================================================
# Helper: evaluate model
# ============================================================
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "AUC":       round(roc_auc_score(y_test, y_prob), 4),
    }
    return metrics, y_pred, y_prob


# ============================================================
# 1. LOAD & PREPROCESS DATA
# ============================================================
# 改动说明：
#   参考代码的数据处理比你原来的强在以下几点：
#   [A] 多删了两列零方差/极高缺失列（STDs:cervical condylomatosis, STDs:AIDS）
#   [B] 用 IQR Winsorization 而不是直接 fillna，对连续变量做异常值截断
#   [C] 先 fillna 再 clip，顺序正确（先补全再截断）
#   这三点共同让特征分布更干净，直接影响 SVC/KNN 的距离计算质量。
# ============================================================
print("=" * 60)
print("LOADING & PREPROCESSING DATA")
print("=" * 60)

df = pd.read_csv("kag_risk_factors_cervical_cancer.csv")
df = df.replace("?", np.nan)

# [A] 删除极高缺失率 + 零方差列
#     参考代码比你多删了 STDs:cervical condylomatosis 和 STDs:AIDS
#     这两列在你的代码里保留着，会给 RFE 引入噪声特征
drop_cols = [
    'STDs: Time since first diagnosis',
    'STDs: Time since last diagnosis',
    'STDs:cervical condylomatosis',   # 新增删除
    'STDs:AIDS'                       # 新增删除
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns], errors='ignore')

# 转为数值型
df = df.apply(pd.to_numeric, errors='coerce')

# 用中位数填充缺失值（中位数比均值对异常值更鲁棒）
df = df.fillna(df.median())

# [B] IQR Winsorization：对连续变量做异常值截断
#     参考代码用 clip() 而不是删行，保留了样本量
#     对 SVC/KNN 这类距离敏感模型非常关键
def replace_outliers_iqr(series):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return series.clip(lower=lower_bound, upper=upper_bound)

continuous_cols = [
    'Age', 'Number of sexual partners',
    'First sexual intercourse', 'Smokes (packs/year)',
    'Num of pregnancies', 'IUD (years)'
]
for col in continuous_cols:
    if col in df.columns:
        df[col] = replace_outliers_iqr(df[col])

# 四个诊断目标（保持你原来的设计）
TARGET_COLS = ["Biopsy", "Hinselmann", "Schiller", "Citology"]

print(f"Dataset shape after cleaning: {df.shape}")
for t in TARGET_COLS:
    print(f"  {t} distribution: {df[t].value_counts().to_dict()}")


# ============================================================
# PHASE 1 — Single Model Baseline (all 4 targets)
# ============================================================
# 改动说明：
#   参考代码的 baseline 用的是纯 SMOTE（不是你原来的 SMOTEENN）。
#   SMOTEENN 会在过采样后再做 ENN 清洗，删掉边界样本，导致训练集变小。
#   baseline 阶段应该用更简单的 SMOTE，让模型尽可能学到足够多的正样本。
#   另外参考代码加了 scale_pos_weight（XGBoost）和 class_weight='balanced'（RF/SVC），
#   这让模型在不平衡数据上的召回率明显提升。
# ============================================================
print("\n" + "=" * 60)
print("PHASE 1: SINGLE MODEL BASELINE")
print("=" * 60)

phase1_results = {}
phase1_roc     = {}

for target in TARGET_COLS:
    print(f"\n  Target: {target}")
    y = df[target]
    X = df.drop(columns=TARGET_COLS, errors='ignore')

    # [C] stratify=y 保证训练/测试集的正负比例一致
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # [D] baseline 用纯 SMOTE（参考代码做法），不用 SMOTEENN
    #     SMOTEENN 留给后续阶段做精细化处理
    smote = SMOTE(random_state=42)
    X_tr_bal, y_tr_bal = smote.fit_resample(X_train_s, y_train)

    # [E] 加了 class_weight='balanced' 和 scale_pos_weight
    #     这是参考代码 baseline 比你高的关键原因之一
    base_models = {
        "SVC":          SVC(probability=True, class_weight='balanced', random_state=42),
        "KNN":          KNeighborsClassifier(),
        "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=42),
        "XGBoost":      XGBClassifier(
                            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
                            eval_metric='logloss', random_state=42
                        ),
    }

    target_results = {}
    target_roc     = {}

    for name, model in base_models.items():
        t0 = time.time()
        model.fit(X_tr_bal, y_tr_bal)
        train_time = round(time.time() - t0, 2)

        metrics, _, y_prob = evaluate_model(model, X_test_s, y_test)
        metrics["Train_Time(s)"] = train_time
        target_results[name] = metrics

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        target_roc[name] = (fpr, tpr, metrics["AUC"])

        print(f"    {name}: Acc={metrics['Accuracy']} Rec={metrics['Recall']} "
              f"F1={metrics['F1']} AUC={metrics['AUC']}")

    phase1_results[target] = pd.DataFrame(target_results).T
    phase1_roc[target]     = target_roc

print("\n  ===== PHASE 1 SUMMARY =====")
for target, df_res in phase1_results.items():
    print(f"\n  [{target}]")
    print(df_res.to_string())


# ============================================================
# PHASE 2 — Feature Selection (RFE + Tree Importance) & SMOTEENN
# ============================================================
print("\n" + "=" * 60)
print("PHASE 2: FEATURE SELECTION (RFE + TREE IMPORTANCE)")
print("=" * 60)

primary_target = "Biopsy"
y_fs = df[primary_target]
X_fs = df.drop(columns=TARGET_COLS, errors='ignore')

X_tr_fs, X_te_fs, y_tr_fs, y_te_fs = train_test_split(
    X_fs, y_fs, test_size=0.2, random_state=42, stratify=y_fs
)

scaler_fs = StandardScaler()
X_tr_fs_s = scaler_fs.fit_transform(X_tr_fs)

# --- RFE with Random Forest estimator ---
rf_rfe = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rfe = RFE(estimator=rf_rfe, n_features_to_select=18, step=1)
rfe.fit(X_tr_fs_s, y_tr_fs)

rfe_selected = X_fs.columns[rfe.support_].tolist()
print(f"\n  RFE selected {len(rfe_selected)} features:")
print(f"  {rfe_selected}")

# --- Tree-based Importance ---
rf_imp = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42)
rf_imp.fit(X_tr_fs_s, y_tr_fs)
importances = pd.Series(rf_imp.feature_importances_, index=X_fs.columns).sort_values(ascending=False)
tree_top18  = importances.head(18).index.tolist()
print(f"\n  Tree-importance top 18 features:")
print(f"  {tree_top18}")

# Union: overlap 优先，再从 tree importance 补充到 18 个
overlap = [f for f in rfe_selected if f in tree_top18]
fill    = [f for f in tree_top18 if f not in overlap]
selected_features = (overlap + fill)[:18]
print(f"\n  Final selected features ({len(selected_features)}): {selected_features}")

plt.figure(figsize=(10, 6))
importances[selected_features].sort_values().plot(kind='barh', color='steelblue')
plt.title("Tree-Based Feature Importance (Selected Features)")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150)
plt.close()
print("  Saved: feature_importance.png")


# ============================================================
# PHASE 3 — Hybrid Soft-Voting Models (all 4 targets)
# ============================================================
# 改动说明：
#   Phase 3 开始换回 SMOTEENN（过采样 + ENN 清洗边界噪声），
#   配合特征选择后更干净的特征集，让混合模型的决策边界更清晰。
# ============================================================
print("\n" + "=" * 60)
print("PHASE 3: HYBRID SOFT-VOTING MODELS")
print("=" * 60)

phase3_results = {}
phase3_roc     = {}

for target in TARGET_COLS:
    print(f"\n  Target: {target}")
    y = df[target]
    X = df[selected_features]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    smote = SMOTEENN(random_state=42)
    X_tr_bal, y_tr_bal = smote.fit_resample(X_train_s, y_train)

    svc_m = SVC(probability=True, class_weight='balanced', random_state=42)
    knn_m = KNeighborsClassifier()
    rf_m  = RandomForestClassifier(class_weight='balanced', random_state=42)
    xgb_m = XGBClassifier(
        scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
        eval_metric='logloss', random_state=42
    )

    hybrid_base_base = VotingClassifier(
        estimators=[('svc', SVC(probability=True, class_weight='balanced', random_state=42)),
                    ('knn', KNeighborsClassifier())],
        voting='soft'
    )
    hybrid_adv_base = VotingClassifier(
        estimators=[('rf',  RandomForestClassifier(class_weight='balanced', random_state=42)),
                    ('svc', SVC(probability=True, class_weight='balanced', random_state=42))],
        voting='soft'
    )
    hybrid_xgb_knn = VotingClassifier(
        estimators=[('xgb', XGBClassifier(
                         scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
                         eval_metric='logloss', random_state=42)),
                    ('knn', KNeighborsClassifier())],
        voting='soft'
    )

    all_models = {
        "SVC":              svc_m,
        "KNN":              knn_m,
        "RandomForest":     rf_m,
        "XGBoost":          xgb_m,
        "Hybrid_SVC+KNN":   hybrid_base_base,
        "Hybrid_RF+SVC":    hybrid_adv_base,
        "Hybrid_XGB+KNN":   hybrid_xgb_knn,
    }

    target_results = {}
    target_roc     = {}

    for name, model in all_models.items():
        t0 = time.time()
        model.fit(X_tr_bal, y_tr_bal)
        train_time = round(time.time() - t0, 2)

        metrics, _, y_prob = evaluate_model(model, X_test_s, y_test)
        metrics["Train_Time(s)"] = train_time
        target_results[name] = metrics

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        target_roc[name] = (fpr, tpr, metrics["AUC"])

        print(f"    {name}: Acc={metrics['Accuracy']} Rec={metrics['Recall']} "
              f"F1={metrics['F1']} AUC={metrics['AUC']}")

    phase3_results[target] = pd.DataFrame(target_results).T
    phase3_roc[target]     = target_roc

print("\n  ===== PHASE 3 SUMMARY =====")
for target, df_res in phase3_results.items():
    print(f"\n  [{target}]")
    print(df_res.to_string())


# ============================================================
# PHASE 4 — Hyperparameter Tuning (GridSearch + RandomSearch)
# ============================================================
print("\n" + "=" * 60)
print("PHASE 4: HYPERPARAMETER TUNING")
print("=" * 60)

target = "Biopsy"
y = df[target]
X = df[selected_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

smote = SMOTEENN(random_state=42)
X_tr_bal, y_tr_bal = smote.fit_resample(X_train_s, y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid Search: SVC
print("\n  [Grid Search] SVC ...")
svc_param_grid = {
    'C':      [0.1, 1, 10],
    'kernel': ['rbf', 'linear'],
    'gamma':  ['scale', 'auto'],
}
svc_gs = GridSearchCV(
    SVC(probability=True, class_weight='balanced', random_state=42),
    svc_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1
)
t0 = time.time()
svc_gs.fit(X_tr_bal, y_tr_bal)
svc_time = round(time.time() - t0, 2)
print(f"    Best params : {svc_gs.best_params_}")
print(f"    Best CV AUC : {round(svc_gs.best_score_, 4)}")
print(f"    Training time: {svc_time}s")

# Random Search: KNN
print("\n  [Random Search] KNN ...")
knn_param_dist = {
    'n_neighbors': [3, 5, 7, 9, 11],
    'weights':     ['uniform', 'distance'],
    'metric':      ['euclidean', 'manhattan'],
}
knn_rs = RandomizedSearchCV(
    KNeighborsClassifier(),
    knn_param_dist, n_iter=12, cv=cv, scoring='roc_auc',
    random_state=42, n_jobs=-1
)
t0 = time.time()
knn_rs.fit(X_tr_bal, y_tr_bal)
knn_time = round(time.time() - t0, 2)
print(f"    Best params : {knn_rs.best_params_}")
print(f"    Best CV AUC : {round(knn_rs.best_score_, 4)}")
print(f"    Training time: {knn_time}s")

# Grid Search: Random Forest
print("\n  [Grid Search] Random Forest ...")
rf_param_grid = {
    'n_estimators':      [100, 200],
    'max_depth':         [None, 10, 20],
    'min_samples_split': [2, 5],
}
rf_gs = GridSearchCV(
    RandomForestClassifier(class_weight='balanced', random_state=42),
    rf_param_grid, cv=cv, scoring='roc_auc', n_jobs=-1
)
t0 = time.time()
rf_gs.fit(X_tr_bal, y_tr_bal)
rf_time = round(time.time() - t0, 2)
print(f"    Best params : {rf_gs.best_params_}")
print(f"    Best CV AUC : {round(rf_gs.best_score_, 4)}")
print(f"    Training time: {rf_time}s")

# Random Search: XGBoost
print("\n  [Random Search] XGBoost ...")
pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
xgb_param_dist = {
    'n_estimators':  [100, 200, 300],
    'max_depth':     [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample':     [0.7, 0.8, 1.0],
}
xgb_rs = RandomizedSearchCV(
    XGBClassifier(scale_pos_weight=pos_weight, eval_metric='logloss', random_state=42),
    xgb_param_dist, n_iter=15, cv=cv, scoring='roc_auc',
    random_state=42, n_jobs=-1
)
t0 = time.time()
xgb_rs.fit(X_tr_bal, y_tr_bal)
xgb_time = round(time.time() - t0, 2)
print(f"    Best params : {xgb_rs.best_params_}")
print(f"    Best CV AUC : {round(xgb_rs.best_score_, 4)}")
print(f"    Training time: {xgb_time}s")

# Optimized Hybrid Models
print("\n  Building optimized hybrid models ...")

opt_hybrid_bb = VotingClassifier(
    estimators=[('svc', svc_gs.best_estimator_),
                ('knn', knn_rs.best_estimator_)],
    voting='soft', weights=[2, 1]
)
opt_hybrid_ab = VotingClassifier(
    estimators=[('rf',  rf_gs.best_estimator_),
                ('svc', svc_gs.best_estimator_)],
    voting='soft', weights=[2, 1]
)
opt_hybrid_xk = VotingClassifier(
    estimators=[('xgb', xgb_rs.best_estimator_),
                ('knn', knn_rs.best_estimator_)],
    voting='soft', weights=[3, 1]
)

tuned_models = {
    "Tuned_SVC":              svc_gs.best_estimator_,
    "Tuned_KNN":              knn_rs.best_estimator_,
    "Tuned_RF":               rf_gs.best_estimator_,
    "Tuned_XGBoost":          xgb_rs.best_estimator_,
    "Tuned_Hybrid_SVC+KNN":   opt_hybrid_bb,
    "Tuned_Hybrid_RF+SVC":    opt_hybrid_ab,
    "Tuned_Hybrid_XGB+KNN":   opt_hybrid_xk,
}

phase4_results = {}
phase4_roc     = {}

for name, model in tuned_models.items():
    t0 = time.time()
    model.fit(X_tr_bal, y_tr_bal)
    train_time = round(time.time() - t0, 2)

    metrics, _, y_prob = evaluate_model(model, X_test_s, y_test)
    metrics["Train_Time(s)"] = train_time
    phase4_results[name] = metrics

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    phase4_roc[name] = (fpr, tpr, metrics["AUC"])

    print(f"    {name}: Acc={metrics['Accuracy']} Rec={metrics['Recall']} "
          f"F1={metrics['F1']} AUC={metrics['AUC']}")

phase4_df = pd.DataFrame(phase4_results).T
print("\n  ===== PHASE 4 SUMMARY =====")
print(phase4_df.to_string())


# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n" + "=" * 60)
print("GENERATING VISUALIZATIONS")
print("=" * 60)

# (A) Phase 1 vs Phase 4 single model comparison
p1_biopsy = phase1_results["Biopsy"][["Accuracy", "Recall", "F1", "AUC"]]
p4_single  = phase4_df[
    phase4_df.index.str.startswith("Tuned_") &
    ~phase4_df.index.str.contains("Hybrid")
][["Accuracy", "Recall", "F1", "AUC"]]

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
p1_biopsy.plot(kind='bar', ax=axes[0], ylim=(0, 1), colormap='Set2', rot=15)
axes[0].set_title("Phase 1 — Single Models (Biopsy, No Tuning)")
axes[0].set_ylabel("Score")
axes[0].legend(loc='lower right')

p4_single.plot(kind='bar', ax=axes[1], ylim=(0, 1), colormap='Set2', rot=15)
axes[1].set_title("Phase 4 — Tuned Single Models (Biopsy)")
axes[1].set_ylabel("Score")
axes[1].legend(loc='lower right')

plt.suptitle("Single Model Performance: Before vs After Tuning", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("phase1_vs_phase4_single.png", dpi=150)
plt.close()
print("  Saved: phase1_vs_phase4_single.png")

# (B) Phase 4 full comparison
fig, ax = plt.subplots(figsize=(14, 6))
phase4_df[["Accuracy", "Recall", "F1", "AUC"]].plot(
    kind='bar', ax=ax, ylim=(0, 1), colormap='tab10', rot=20
)
ax.set_title("Phase 4 — Tuned Single vs Hybrid Models (Biopsy)", fontsize=13, fontweight='bold')
ax.set_ylabel("Score")
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig("phase4_full_comparison.png", dpi=150)
plt.close()
print("  Saved: phase4_full_comparison.png")

# (C) ROC Curves — Phase 4
plt.figure(figsize=(10, 7))
for name, (fpr, tpr, auc) in phase4_roc.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc})")
plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Tuned Models (Biopsy)", fontweight='bold')
plt.legend(fontsize=8, loc='lower right')
plt.tight_layout()
plt.savefig("roc_curve_phase4.png", dpi=150)
plt.close()
print("  Saved: roc_curve_phase4.png")

# (D) ROC Curves per target (Phase 3 hybrid)
hybrid_names = ["Hybrid_SVC+KNN", "Hybrid_RF+SVC", "Hybrid_XGB+KNN"]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, target in zip(axes.flat, TARGET_COLS):
    for name in hybrid_names:
        if name in phase3_roc[target]:
            fpr, tpr, auc = phase3_roc[target][name]
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc})")
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_title(f"ROC — {target}")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=7)
plt.suptitle("Phase 3 Hybrid Model ROC Curves (All Targets)", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("roc_all_targets_hybrid.png", dpi=150)
plt.close()
print("  Saved: roc_all_targets_hybrid.png")

# (E) Confusion Matrix — best Phase 4 model
best_name  = phase4_df["AUC"].idxmax()
best_model = tuned_models[best_name]
y_prob_best = best_model.predict_proba(X_test_s)[:, 1]
y_pred_best = (y_prob_best > 0.3).astype(int)
cm = confusion_matrix(y_test, y_pred_best)

plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Cancer', 'Cancer'],
            yticklabels=['No Cancer', 'Cancer'])
plt.title(f"Confusion Matrix — {best_name} (Biopsy)", fontweight='bold')
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix_best.png", dpi=150)
plt.close()
print(f"  Saved: confusion_matrix_best.png  (Best model: {best_name})")

# (F) Multi-target heatmap (Phase 3)
recall_table = {}
auc_table    = {}
for target in TARGET_COLS:
    recall_table[target] = {k: v["Recall"] for k, v in phase3_results[target].iterrows()}
    auc_table[target]    = {k: v["AUC"]    for k, v in phase3_results[target].iterrows()}

recall_df = pd.DataFrame(recall_table)
auc_df    = pd.DataFrame(auc_table)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.heatmap(recall_df, annot=True, fmt=".2f", cmap="YlOrRd",
            ax=axes[0], linewidths=0.5, vmin=0, vmax=1)
axes[0].set_title("Recall Across All Targets (Phase 3)")
sns.heatmap(auc_df, annot=True, fmt=".2f", cmap="Blues",
            ax=axes[1], linewidths=0.5, vmin=0, vmax=1)
axes[1].set_title("AUC Across All Targets (Phase 3)")
plt.suptitle("Model Performance Heatmap — All Diagnostic Targets",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig("heatmap_all_targets.png", dpi=150)
plt.close()
print("  Saved: heatmap_all_targets.png")


# ============================================================
# SHAP — Interpretability
# ============================================================
if SHAP_AVAILABLE:
    print("\n" + "=" * 60)
    print("SHAP INTERPRETABILITY ANALYSIS")
    print("=" * 60)

    shap_model_name = "Tuned_RF" if "Tuned_RF" in phase4_results else "Tuned_XGBoost"
    shap_model = tuned_models[shap_model_name]

    if isinstance(shap_model, VotingClassifier):
        for label, est in shap_model.named_estimators_.items():
            if isinstance(est, (RandomForestClassifier, XGBClassifier)):
                shap_model = est
                shap_model_name += f"({label})"
                break

    print(f"\n  Using model for SHAP: {shap_model_name}")

    explainer   = shap.TreeExplainer(shap_model)
    X_test_df   = pd.DataFrame(X_test_s, columns=selected_features)
    shap_values = explainer.shap_values(X_test_df)

    sv = shap_values[1] if isinstance(shap_values, list) else shap_values

    plt.figure()
    shap.summary_plot(sv, X_test_df, show=False, max_display=18)
    plt.title(f"SHAP Summary Plot — {shap_model_name}", fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: shap_summary.png")

    plt.figure()
    shap.summary_plot(sv, X_test_df, plot_type='bar', show=False, max_display=18)
    plt.title(f"SHAP Feature Importance — {shap_model_name}", fontweight='bold')
    plt.tight_layout()
    plt.savefig("shap_importance_bar.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: shap_importance_bar.png")

    # Force plot with error handling
    try:
        plt.figure()
        shap.plots.force(
            explainer.expected_value if not isinstance(explainer.expected_value, list)
            else explainer.expected_value[1],
            sv,
            X_test_df,
            matplotlib=True,
            show=False
        )
        plt.savefig("shap_force_plot_sample.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("  Saved: shap_force_plot_sample.png")
    except Exception as e:
        print(f"  Skipped force plot (error: {type(e).__name__})")
else:
    print("\nSHAP ANALYSIS SKIPPED (pip install shap to enable)")


# ============================================================
# FINAL SUMMARY TABLE
# ============================================================
print("\n" + "=" * 60)
print("FINAL SUMMARY — ALL PHASES (Biopsy Target)")
print("=" * 60)

summary_rows = {}

for model_name, row in phase1_results["Biopsy"].iterrows():
    summary_rows[f"[P1] {model_name}"] = row[["Accuracy", "Recall", "F1", "AUC"]]

for model_name, row in phase3_results["Biopsy"].iterrows():
    if "Hybrid" in model_name:
        summary_rows[f"[P3] {model_name}"] = row[["Accuracy", "Recall", "F1", "AUC"]]

for model_name, row in phase4_df.iterrows():
    summary_rows[f"[P4] {model_name}"] = row[["Accuracy", "Recall", "F1", "AUC"]]

final_summary = pd.DataFrame(summary_rows).T
print(final_summary.to_string())
final_summary.to_csv("final_summary.csv")
print("\n  Saved: final_summary.csv")

print("\n" + "=" * 60)
print("PIPELINE COMPLETE")
print("=" * 60)