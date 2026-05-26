# prompts.py
"""
All LLM prompt templates. Every node that calls an LLM uses one of these.
Keep them structured — LLMs follow structure better than prose paragraphs.

Changes vs prior version:
- MODEL_SELECTION_PROMPT significantly trimmed:
    * Removed verbose prose instructions (5 bullet points → 3 concise directives)
    * Profile injection now uses slim_profile dict (no dtypes/skewness blobs)
    * Task injection uses slim_task dict (3 fields only)
    * Removed freeform "Consider:" section — replaced with tighter constraints
  Result: ~40% smaller prompt, faster Ollama completion, less hallucination surface.
"""

TASK_HEURISTICS = """
Task type detection rules:
- IF target has exactly 2 unique values → binary_classification
- IF target has 3-20 unique values AND categorical dtype → multiclass_classification
- IF target is float/int with many unique values (>20% of rows) → regression
- IF datetime index OR datetime column present with trend → time_series_forecasting
- IF no target found AND dataset has strong temporal structure → time_series_forecasting
- IF no target found, infer the most likely supervised target; if temporal structure is dominant → time_series_forecasting
Target leakage: columns that represent post-outcome events (approval_date, outcome_code, churn_reason).
"""

LEAKAGE_PATTERNS = """
Leakage pattern types:
1. TIME-BASED: columns that exist only after the outcome event (e.g., cancellation_date when target=churn)
2. LABEL-DERIVED: columns clearly derived from the target (e.g., fraud_review_notes when target=fraud)
3. NEAR-DUPLICATE: columns that are perfect predictors because of data collection (e.g., loan_amount=0 for all rejections)
4. FUTURE INFO: aggregated statistics computed over the full period including future rows
Flag any column with strong keyword match to outcome events or that seems to contain outcome information.
"""

TASK_DETECTION_PROMPT = """
You are a senior data scientist performing initial dataset assessment.

DATASET PROFILE:
{profile}

HEURISTICS TO APPLY:
{heuristics}

LEAKAGE PATTERNS TO CHECK:
{leakage_patterns}

RAG CONTEXT (domain knowledge from similar past datasets):
{rag_context}

USER CONTEXT (optional hint): {user_context}

Your job:
1. Determine the ML task type using the heuristics above
2. Identify the most likely target column (look for: 'target', 'label', 'churn', 'price', 'survived', 'y', last column)
3. List any columns that look like target leakage with your reasoning
4. Assign a confidence score 0.0-1.0 for your task detection
5. Identify the domain if detectable from column names
6. Recommend primary metrics and validation strategy

Return a fully structured response. Be decisive — pick the most likely task type.
"""


EDA_INTERPRETATION_PROMPT = """
Return concise structured reasoning only. Avoid long explanations. Be direct.

You are a data scientist interpreting computed statistics. Your job is to EXPLAIN, not compute.
All numbers below were computed by pandas — do not recalculate anything.

COMPUTED STATISTICS:
{eda_stats}

TASK TYPE: {task_type}
TARGET COLUMN: {target_column}
DATASET PROFILE SUMMARY: {profile_summary}

Based on these pre-computed statistics, provide:

1. primary_insight: The single most important pattern or problem you see
2. data_quality_issues: Specific quality problems that must be fixed (missing values, outliers, imbalance)
3. modeling_implications: What these patterns mean for model and metric choices
4. recommended_preprocessing: Concrete preprocessing steps implied by these findings

Be specific. Reference the actual numbers from the statistics above.
"""


PREPROCESSING_PROMPT = """
You are a preprocessing specialist reviewing columns to drop.
Rule-based imputation and encoding have already been decided — your ONLY job is columns_to_drop.

DATASET PROFILE:
{profile}

TASK INFORMATION:
{task}

COLUMNS STILL IN DATASET (after leakage and null drops):
{remaining_columns}

RAG KNOWLEDGE (best practices):
{rag_context}

RETRY CONTEXT (read carefully if this is a retry):
{retry_context}

COLUMNS TO DROP — flag any of:
- ID or key columns that weren't already removed (e.g. account_id, record_key)
- Near-constant columns (>95% same value)
- Columns with name/content strongly suggesting they encode the target outcome
- Columns that are redundant duplicates of another column

Return columns_to_drop as a list. Return an empty list if nothing should be dropped.
Set all other fields (imputation_strategy, encoding_strategy, scaling_strategy, imbalance_strategy)
to their defaults — they are ignored by the caller.

IMPORTANT COLUMN DROP RULES:
Do NOT drop columns that appear in the RAG/domain context as relevant predictors.
Only suggest dropping:
1. True ID columns (customerID, uuid, row_number)
2. Near-constant columns (>95% same value)
3. Exact duplicate columns
4. Confirmed leakage columns (already handled before this step)
5. Columns with >80% nulls

When uncertain → KEEP the column.
Never drop based on vague reasoning like "redundant", "low value", or "probably unnecessary".

Provide a brief reasoning field.
"""


FEATURE_ENG_PROMPT = """
Return concise structured output only. No long explanations. Maximum 1 sentence per feature reasoning.

You are a feature engineering expert. Suggest useful new features for this dataset.

TASK: {task}
AVAILABLE COLUMNS: {available_columns}
NUMERIC COLUMNS: {numeric_columns}
HIGH CORRELATIONS (don't create more of these pairs): {high_correlations}

RAG DOMAIN KNOWLEDGE:
{rag_context}

Suggest up to 5 new features using only these operations:
- ratio: col_a / col_b (e.g., debt_to_income_ratio)
- product: col_a * col_b (e.g., price * quantity)
- difference: col_a - col_b (e.g., end_date - start_date)
- log: log(col + 1) for skewed columns (e.g., log_income)
- bin: quantile-based binning of a numeric column into N groups

IMPORTANT: source_columns must be EXACT column names from the available_columns list provided.
Do NOT invent column names. Do NOT use the interaction operation (it is disabled).

Rules:
- Only suggest features when there's a genuine business/statistical reason
- Source columns must be in the available columns list
- Names must be descriptive snake_case
- Do NOT suggest features that would create leakage

Also list any existing features worth dropping (near-perfect correlation with another feature, ID columns, etc.)
"""


MODEL_SELECTION_PROMPT = """
You are a model selection expert. Pick the best model from the candidates below.

HEURISTIC-RANKED CANDIDATES (higher score = better fit for this dataset):
{candidates}

DATASET SUMMARY:
{profile}

DATASET CHARACTERISTICS

HIGH-CARDINALITY COLUMNS (>50 unique values, use ONLY these for cardinality reasoning):
{high_cardinality_columns}

Do NOT infer cardinality for any column not listed above.

TASK: {task}

BASELINE METRICS (LogisticRegression/Ridge — model must beat this meaningfully):
{baseline_metrics}

PRIOR EXPERIMENTS FROM RAG (similar datasets, if any):
{rag_prior_experiments}

RULES:
1. Confirm or override the heuristic ranking based on dataset characteristics.
2. The chosen_model MUST be one of the names in the candidates list — no other values are valid.
3. high_cardinality_cols in the profile are exact computed counts — use them as-is, do not estimate.
4. If prior RAG experiments exist for this domain/task, weight them heavily.
5. Keep reasoning concise — 2-3 sentences maximum.

Return: chosen_model, reasoning (2-3 sentences), expected_improvement_over_baseline.
"""


SHAP_NARRATION_PROMPT = """
You are a data science communicator explaining model predictions to a non-technical audience.

TOP FEATURES BY SHAP IMPORTANCE (feature name, mean absolute SHAP value):
{top_features}

CONTEXT:
- Task: {task_type}
- Target column: {target_column}
- Domain: {domain}
- Model used: {model_name}
- Model performance: {metrics}

Write:
1. narration: A 2-3 sentence plain-English summary of what drives predictions.
   Start with the most important insight. Avoid terms like "SHAP", "gradient", "feature importance".
2. top_feature_explanations: One sentence per top-3 feature explaining what it means in real-world terms.
3. actionable_insights: 2-3 insights a business person could act on based on these findings.

Write for a business audience. Mention the domain. Be concrete and specific.
"""
