# ingest_docs.py
"""
Run this once before your first pipeline run to populate the knowledge base.
Usage:  python ingest_docs.py
Re-run to refresh (uses upsert, safe to run multiple times).
"""
from rag import get_collection

# ── Knowledge documents with metadata tags ────────────────────────────────────
# category must match the node_name used in retrieve_for_node() calls
DOCUMENTS = [

    # ── Preprocessing best practices ──────────────────────────────────────
    {
        "text": (
            "For binary classification with class imbalance ratio >3:1, apply SMOTE "
            "on training data only (after split), or use class_weight='balanced'. "
            "Never use accuracy as primary metric on imbalanced datasets — use F1, "
            "AUC-ROC, or Average Precision Score instead."
        ),
        "meta": {"category": "preprocessing", "task_type": "binary_classification", "domain": "general"},
    },
    {
        "text": (
            "High cardinality categoricals (>50 unique values): use CatBoost which handles "
            "them natively without any encoding, or apply frequency encoding. "
            "One-hot encoding on high-cardinality columns causes dimensionality explosion "
            "and degrades tree model performance."
        ),
        "meta": {"category": "preprocessing", "task_type": "all", "domain": "general"},
    },
    {
        "text": (
            "Time series train-test split: NEVER use random split. Always split "
            "chronologically — train on past rows, validate on future rows. "
            "Random splitting causes data leakage by letting the model see future patterns."
        ),
        "meta": {"category": "preprocessing", "task_type": "time_series_forecasting", "domain": "general"},
    },
    {
        "text": (
            "Missing value imputation: <5% missing → median (numeric) or mode (categorical). "
            "5-30% missing → KNN imputation (numeric) or 'Unknown' category (categorical). "
            ">30% missing → drop the column unless it is domain-critical. "
            "Always add a binary indicator flag when imputing >5% of a column."
        ),
        "meta": {"category": "preprocessing", "task_type": "all", "domain": "general"},
    },
    {
        "text": (
            "Scaling rule: StandardScaler or RobustScaler for linear models, SVM, neural nets. "
            "Tree-based models (XGBoost, LightGBM, CatBoost, RandomForest) are scale-invariant — "
            "scaling them wastes time and adds no value. "
            "Use RobustScaler (median/IQR based) when data has heavy outliers."
        ),
        "meta": {"category": "preprocessing", "task_type": "all", "domain": "general"},
    },

    # ── Model selection heuristics ────────────────────────────────────────
    {
        "text": (
            "LightGBM vs XGBoost for large datasets (>100k rows): LightGBM is 3-5x faster "
            "due to its histogram algorithm. Both achieve similar accuracy. "
            "Default to LightGBM when training speed is a constraint."
        ),
        "meta": {"category": "model_selection", "task_type": "all", "domain": "general"},
    },
    {
        "text": (
            "CatBoost excels when data has many categorical columns, especially high-cardinality ones. "
            "It handles NaN natively and needs minimal preprocessing. "
            "Slightly slower than LightGBM but often most accurate on categorical-heavy tabular data."
        ),
        "meta": {"category": "model_selection", "task_type": "all", "domain": "general"},
    },
    {
        "text": (
            "Medical classification (disease prediction, hospital readmission): "
            "Logistic regression provides interpretable coefficients and is baseline. "
            "Gradient boosting (XGBoost/LightGBM) achieves better AUC. "
            "Always report precision and recall separately — in medical contexts false negatives cost more."
        ),
        "meta": {"category": "model_selection", "task_type": "binary_classification", "domain": "medical"},
    },
    {
        "text": (
            "Financial fraud detection: extreme class imbalance (0.1-1% fraud rate) is typical. "
            "Primary metric must be Average Precision Score, not AUC-ROC (which is misleadingly high). "
            "SMOTE + XGBoost with scale_pos_weight is a strong baseline. "
            "Threshold tuning post-training for precision/recall trade-off is essential."
        ),
        "meta": {"category": "model_selection", "task_type": "binary_classification", "domain": "financial"},
    },
    {
        "text": (
            "Telecom churn prediction: CatBoost and XGBoost consistently outperform Random Forest. "
            "Target AUC-ROC >0.85 is achievable with good features. "
            "Contract type (month-to-month), tenure, and monthly charges are historically top predictors. "
            "Apply class_weight='balanced' or SMOTE — churn is typically 15-30% of customers."
        ),
        "meta": {"category": "model_selection", "task_type": "binary_classification", "domain": "telco"},
    },
    {
        "text": (
            "Regression with heavy outlier targets (e.g., house prices, salary): "
            "Log-transform the target variable if it is right-skewed (skewness >1.5). "
            "Report metrics in original scale by inverse-transforming predictions. "
            "MAE is more robust than RMSE as primary metric when outliers exist."
        ),
        "meta": {"category": "model_selection", "task_type": "regression", "domain": "general"},
    },

    # ── Feature engineering patterns ──────────────────────────────────────
    {
        "text": (
            "Time series features: create lag features at lags 1, 7, 14, 30 of the target. "
            "Rolling statistics (rolling_mean_7d, rolling_std_30d) capture trend and volatility. "
            "Seasonal features (day_of_week, month, is_weekend, is_holiday) often boost performance significantly. "
            "Always apply lag features before split to avoid NaN leakage."
        ),
        "meta": {"category": "feature_engineering", "task_type": "time_series_forecasting", "domain": "general"},
    },
    {
        "text": (
            "Financial dataset features: ratio features are powerful predictors — "
            "debt_to_income_ratio, credit_utilization_ratio, loan_to_value. "
            "Interaction features between spending categories often reveal risk patterns. "
            "Customer tenure binned into quartiles often performs better than raw tenure."
        ),
        "meta": {"category": "feature_engineering", "task_type": "all", "domain": "financial"},
    },
    {
        "text": (
            "HR / employee attrition: tenure-based features (years_at_company, "
            "years_since_last_promotion) are strong predictors. "
            "Salary percentile within department is more informative than raw salary. "
            "Department and job role interaction features often reveal attrition clusters."
        ),
        "meta": {"category": "feature_engineering", "task_type": "binary_classification", "domain": "hr"},
    },
    {
        "text": (
            "Target encoding for high-cardinality categoricals: must be computed with "
            "k-fold cross-validation on training data to prevent leakage. "
            "Never compute target mean on the full training set and use directly — "
            "this inflates validation scores by 2-5% and creates a false impression of model quality."
        ),
        "meta": {"category": "feature_engineering", "task_type": "all", "domain": "general"},
    },

    # ── Evaluation guides ─────────────────────────────────────────────────
    {
        "text": (
            "Binary classification metric guide: "
            "ROC-AUC = ability to rank positive over negative (threshold-independent). "
            "Average Precision = area under precision-recall curve (better for imbalanced). "
            "F1-weighted = harmonic mean of precision and recall (threshold-dependent). "
            "Balanced Accuracy = mean recall per class (robust to imbalance). "
            "For imbalanced data: Average Precision is the most honest primary metric."
        ),
        "meta": {"category": "evaluation", "task_type": "binary_classification", "domain": "general"},
    },
    {
        "text": (
            "Regression metrics: R² measures explained variance (0=mean baseline, 1=perfect, negative=worse than mean). "
            "RMSE penalizes large errors quadratically (sensitive to outliers). "
            "MAE is linear penalty (robust to outliers). "
            "Report all three. If target is log-transformed, inverse-transform predictions before reporting."
        ),
        "meta": {"category": "evaluation", "task_type": "regression", "domain": "general"},
    },

    # ── Known failure patterns for critic node ────────────────────────────
    {
        "text": (
            "Overfitting pattern: train_score - val_score > 0.15 signals overfitting. "
            "Fixes: reduce max_depth (XGBoost: 4-5, LightGBM: 15-25 leaves), "
            "increase min_child_weight/min_samples_leaf, add L2 regularization, "
            "reduce n_estimators with early stopping, or apply feature selection."
        ),
        "meta": {"category": "critic", "task_type": "all", "domain": "general"},
    },
    {
        "text": (
            "Target leakage pattern: accuracy >98% on binary classification or R² >0.98 = suspect leakage. "
            "Check: future-derived columns (dates/timestamps after the outcome), "
            "label-derived columns (reason codes, outcome descriptions), "
            "ID columns with accidental correlation to target, "
            "aggregated stats computed over full dataset including test rows."
        ),
        "meta": {"category": "critic", "task_type": "all", "domain": "general"},
    },
    {
        "text": (
            "Class imbalance ignored signal: high accuracy + low F1 gap >0.10. "
            "Example: 92% accuracy, 0.71 F1 on 90/10 split = model predicts majority class. "
            "Fix: apply SMOTE on training set, or set class_weight='balanced', "
            "retrain, and re-evaluate with balanced_accuracy and f1_weighted."
        ),
        "meta": {"category": "critic", "task_type": "binary_classification", "domain": "general"},
    },
]


def ingest(cache_dir: str = "cache") -> None:
    collection = get_collection(cache_dir)

    texts = [d["text"] for d in DOCUMENTS]
    metadatas = [d["meta"] for d in DOCUMENTS]
    ids = [f"doc_{i:04d}" for i in range(len(DOCUMENTS))]

    collection.upsert(documents=texts, metadatas=metadatas, ids=ids)

    print(f"✅ Ingested {len(DOCUMENTS)} documents.")
    print(f"   Total in collection: {collection.count()}")


if __name__ == "__main__":
    ingest()
