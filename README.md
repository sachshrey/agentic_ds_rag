# Agentic Data Science RAG System

> An end-to-end machine learning pipeline where an LLM-guided LangGraph agent profiles a dataset, detects the task, preprocesses features, selects and trains models, critiques the result, explains predictions with SHAP, and writes a full audit report.

---

## What This Project Does

This is not a one-off notebook. It is an **agentic data science workflow** built as a state machine. Each pipeline stage is a LangGraph node, and every decision is recorded into a final Markdown report.

The system combines deterministic ML rules with LLM reasoning:

- **GPT-4.1** for high-stakes reasoning nodes such as task detection, critique, explainability, and reporting
- **Ollama `llama3.2:3b`** for local lower-cost reasoning support
- **ChromaDB RAG** for ML guidance and prior-experiment memory
- **MLflow** for experiment tracking
- **SHAP** for model explainability
- **Pydantic structured outputs** for safer LLM responses

```text
Dataset
  |
  v
Profiler -> Size Router -> Task Detector
                              |
                              v
                             EDA
                              |
                              v
                       Preprocessing <----- Critic retry loop
                              |
                              v
                    Feature Engineering
                              |
                              v
                           Baseline
                              |
                              v
                      Model Selection  <--- RAG context from ChromaDB
                              |
                              v
                           Training
                              |
                              v
                         Evaluation
                              |
                              v
                           Critic
                              |
                              v
                    SHAP Explainer + Report
```

---

## Why It Is Different

| Basic ML scripts | This system |
|---|---|
| Manually inspect and clean a dataset | Profiles the dataset and routes the workflow automatically |
| Pick one model directly | Trains a baseline first, then selects among stronger candidates |
| Report a single accuracy number | Compares against baseline and reports multiple task-aware metrics |
| Mistakes remain hidden | Critic node checks leakage, overfitting, imbalance, poor baseline improvement, and bad regression behavior |
| No reasoning trail | Every node appends decisions to a full audit log |
| No memory across runs | Stores experiment summaries back into ChromaDB for future model-selection context |
| Hard to explain predictions | Produces SHAP plots plus plain-English explanation and actions |

---

## Verified Telco Churn Run

The system was run end-to-end on the IBM Telco Churn dataset.

```bash
python ingest_docs.py

python main.py data/telco_churn.csv \
  --context "Telecom customer churn prediction. Target column is Churn." \
  --output-dir outputs/telco_churn_rag_run
```

### Final Result

| Item | Value |
|---|---|
| Task detected | Binary classification |
| Target column | `Churn` |
| Domain | Telco |
| Model selected | CatBoost |
| Retries triggered | 0 |
| Final feature count | 29 |

### Metrics

| Metric | Score |
|---|---:|
| Accuracy | 0.7466 |
| Balanced Accuracy | 0.7661 |
| F1-Weighted | 0.7602 |
| ROC-AUC | 0.8463 |
| Average Precision | 0.6634 |

### Top SHAP Features

| Rank | Feature | Mean \|SHAP\| |
|---:|---|---:|
| 1 | `tenure` | 0.4830 |
| 2 | `Contract_Two year` | 0.3820 |
| 3 | `InternetService_Fiber optic` | 0.3381 |
| 4 | `PaymentMethod_Electronic check` | 0.2471 |
| 5 | `MonthlyCharges` | 0.1913 |
| 9 | `OnlineSecurity_Yes` | 0.1123 |
| 10 | `TechSupport_Yes` | 0.0768 |

The report correctly handles protected categorical columns after one-hot encoding. For example, `Contract`, `OnlineSecurity`, and `TechSupport` are treated as retained because they appear as encoded features such as `Contract_Two year`, `OnlineSecurity_Yes`, and `TechSupport_Yes`.

---

## Project Structure

```text
.
├── main.py                 # CLI entry point
├── graph.py                # LangGraph state machine
├── state.py                # AgentState Pydantic model
├── schemas.py              # Structured LLM output schemas
├── prompts.py              # Prompt templates
├── rag.py                  # ChromaDB retrieval and experiment memory
├── ingest_docs.py          # Builds the RAG knowledge base
├── checkpointing.py        # Lightweight JSON sidecar checkpoints
├── config.yaml             # Pipeline thresholds and paths
├── requirements.txt        # Python dependencies
├── data/
│   ├── sample_binary_classification.csv
│   └── telco_churn.csv
├── nodes/
│   ├── profiler.py
│   ├── size_router.py
│   ├── task_detector.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── baseline.py
│   ├── model_selection.py
│   ├── training.py
│   ├── evaluation.py
│   ├── critic.py
│   ├── explainer.py
│   └── report.py
└── utils/
    ├── config.py
    ├── llm.py
    └── caching.py
```

---

## Setup

### 1. Clone

```bash
git clone git@github.com:sachshrey/agentic_ds_rag.git
cd agentic_ds_rag
```

### 2. Create Environment

Python 3.12 is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure Secrets

```bash
cp .env.example .env
```

Edit `.env` and set:

```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1
OLLAMA_BASE_URL=http://localhost:11434
EMBED_DEVICE=cpu
```

Note: the current code uses `llama3.2:3b` in `utils/llm.py` for Ollama-backed nodes.

### 4. Install Ollama Model

```bash
ollama pull llama3.2:3b
```

Make sure Ollama is running before executing the pipeline.

### 5. Build RAG Knowledge Base

```bash
python ingest_docs.py
```

This loads ML guidance documents into ChromaDB. If the knowledge base is empty, retrieval degrades gracefully, but model selection is better after ingestion.

### 6. Run Pipeline

```bash
python main.py data/telco_churn.csv \
  --context "Telecom customer churn prediction. Target column is Churn." \
  --output-dir outputs/telco_churn_rag_run
```

Disable cached LLM responses when you want fresh reasoning:

```bash
python main.py data/telco_churn.csv --no-cache
```

---

## Outputs

Each run writes artifacts under the selected output directory:

| File | Purpose |
|---|---|
| `final_report.md` | Complete analysis report and reasoning audit trail |
| `profile_report.html` | Interactive ydata-profiling report |
| `target_distribution.png` | Target distribution plot |
| `correlation_heatmap.png` | Feature correlation plot |
| `shap_summary.png` | Global SHAP feature importance |
| `shap_waterfall.png` | Individual prediction explanation |
| `checkpoints/*.json` | Lightweight per-node state snapshots |

MLflow artifacts and metrics are stored under `mlruns/`.

```bash
mlflow ui --backend-store-uri mlruns --port 5000
```

Open:

```text
http://localhost:5000
```

---

## Supported Workflows

| Task Type | Baseline | Candidate Models | Main Metrics |
|---|---|---|---|
| Binary classification | Logistic Regression | CatBoost, XGBoost, LightGBM, Random Forest, Logistic Regression | F1, ROC-AUC, Average Precision |
| Multiclass classification | Logistic Regression | CatBoost, XGBoost, LightGBM, Random Forest | F1, Balanced Accuracy |
| Regression | Ridge | CatBoost, XGBoost, LightGBM, Random Forest, Ridge | R2, RMSE, MAE |
| Time series forecasting | Ridge | LightGBM, XGBoost | R2, RMSE, MAE |

---

## Reliability Features

- Rule-based preprocessing for imputation, encoding, and safe dropping
- Protected domain columns to prevent accidental removal of important business features
- Encoded-aware protected-column reporting
- Baseline comparison before accepting advanced model value
- Critic loop for high-severity issues
- Retry-aware preprocessing prompt context
- SQLite LLM response cache with `--no-cache` escape hatch
- JSON sidecar checkpoints after each node
- ChromaDB experiment memory for future runs

---

## Repository Hygiene

The repository intentionally excludes generated and local-only files:

```text
.env
.venv/
cache/
outputs/
mlruns/
catboost_info/
__pycache__/
.DS_Store
```

This keeps GitHub focused on source code, configuration, reproducible sample data, and documentation.

---

## Limitations

- Designed for tabular and basic time-series datasets.
- NLP, image, clustering, and anomaly-detection workflows are not first-class paths.
- Ollama speed depends on local hardware.
- OpenAI API access is required for high-stakes nodes.
- The current Ollama model is hardcoded to `llama3.2:3b` in `utils/llm.py`.

---

## License

MIT
