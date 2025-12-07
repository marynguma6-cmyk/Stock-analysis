# Stock Market Trend Prediction - AI Agent Guidelines

## Project Overview
ML pipeline for predicting stock market trends (uptrend/downtrend/sideways) using technical indicators. Three-phase architecture: **data loading/cleaning** → **exploratory analysis** → **model development with MLflow tracking**.

## Key Architecture & Data Flow

### Data Pipeline
1. **Source Data** (`01_data_loading.ipynb`): Loads company info, market indices, and raw stock prices with technical indicators
2. **Cleaning** (`02_data_modelling.ipynb`): Removes NaNs (~10%), applies IQR-based outlier clipping, extracts temporal features (year/month/quarter)
3. **Merged Dataset** (`merged_stock_data.csv`): Joins stock prices with company metadata (sector, IPO date)
4. **Features**: 34+ technical indicators (SMA, EMA, MACD, RSI, Bollinger Bands, ATR, volatility metrics)
5. **Target**: `trend_label` (uptrend/Downtrend/sideways) - categorical, NOT future_return_5d

### Model Development (`mc.ipynb` + `mc_learning.py`)
- **MLflow tracking**: Uses local `mlruns/` directory; models logged with parameters, metrics, and tags
- **Models tested**: Logistic Regression (baseline), Random Forest (optimized with GridSearchCV)
- **Top predictive features** (from feature importance): sma_200, atr_14, volume_sma_20
- **Test strategy**: 70/30 train-test split, 3-fold cross-validation for hyperparameter tuning
- **Saved artifacts**: `optimized_random_forest_model.pkl` (best model from GridSearch)

## Project-Specific Conventions

### File Naming & Locations
- Notebooks: `{01,02}_description.ipynb` (sequenced execution order)
- Data files: CSV format in project root (cleaned_data.csv, merged_stock_data.csv)
- Models: `.pkl` format in project root (joblib/pickle)
- MLflow runs: `mlruns/{experiment_id}/{run_id}/`

### Data Processing Quirks
- **Date handling**: Convert all date columns to datetime immediately, sort by date, check for business-day gaps
- **Target column**: `trend_label` has three classes; ensure multi-class metrics (F1 with `average='weighted'`)
- **Outlier removal**: IQR method (1.5× IQR) applied to 34 affected columns after NaN removal
- **Feature engineering**: Temporal features (year, month, quarter) extracted; no categorical encoding beyond date

### Model Training Specifics
- **No feature scaling for Random Forest** (tree-based); always scale for Logistic Regression
- **Random Forest params to tune**: n_estimators [50-200], max_depth [10-30], min_samples_split [2-10]
- **Evaluation metrics**: accuracy + F1-weighted + precision + recall (not ROC/AUC for multi-class)
- **Baseline comparison**: Logistic Regression accuracy typically ~35-40%; Random Forest ~50-60%

## Critical Developer Workflows

### MLflow Integration
```bash
# View all logged models and metrics:
mlflow ui  # Opens http://localhost:5000
```
- **Experiment name**: "Stock Market Trend Prediction"
- **Run naming**: descriptive (e.g., "logistic_regression_baseline", "random_forest_production")
- **Tagging convention**: model_phase, model_type, status (baseline/production_candidate)
- **Log structure**: parameters → metrics → artifacts (plots/models) → tags

### Model Evaluation Workflow
1. Load cleaned_data.csv (handles missing values and outliers pre-processed)
2. Select numeric features, drop future_return_5d, remove rows where trend_label is null
3. Split with test_size=0.3, random_state=42 (for reproducibility)
4. Scale features (LR only), train, evaluate on test set
5. Log to MLflow with all parameters and metrics

### Running Notebooks Sequentially
- Execute in order: `01_data_loading` → `02_data_modelling` → `mc.ipynb`
- Each notebook reloads data from CSV (not dependent on kernel state)
- `mc_learning.py` is standalone script version (use for batch model training)

## Integration Points & External Dependencies

### Key Libraries & Versions
- **scikit-learn**: LogisticRegression, RandomForestClassifier, GridSearchCV, metrics, preprocessing
- **MLflow**: Logging, experiment tracking, model persistence
- **pandas/numpy**: Data manipulation
- **matplotlib/seaborn**: Visualization (always use plt.tight_layout(), plt.close() for saves)
- **joblib**: Model serialization (preferred over pickle for sklearn models)

### Cross-Component Dependencies
- `load_test_data()` in `mc_learning.py` currently returns dummy data—replace with actual cleaned_data.csv preprocessing
- `plot_feature_importance()` only works with models having `feature_importances_` (trees, forests; not LR)
- Evaluation expects 3-class target; update metrics if task changes to binary classification

## Common Pitfalls & Patterns

### Data Handling
- ❌ Using `future_return_5d` as a feature (it's auxiliary, not predictive for trend)
- ✅ Drop it explicitly: `df.drop(['future_return_5d'], axis=1, errors='ignore')`
- ❌ Forgetting to remove null rows from both X and y simultaneously
- ✅ Use: `mask = ~y.isnull(); X = X[mask]; y = y[mask]`

### Model Development
- ❌ Scaling features before Random Forest
- ✅ Only scale for Logistic Regression/SVM; tree models don't need it
- ❌ Forgetting random_state for reproducibility
- ✅ Always set random_state=42 in train_test_split and model constructors

### Visualization & Logging
- ❌ Leaving plots open in loops (memory leak)
- ✅ Use `plt.close()` after saving
- ❌ Logging raw predictions without metrics context
- ✅ Log parameters, metrics, artifacts together with descriptive run names

## Key Files to Reference

| File | Purpose |
|------|---------|
| `cleaned_data.csv` | Pre-processed features (no NaNs, outliers clipped) |
| `merged_stock_data.csv` | Stock data + company metadata for sector analysis |
| `optimized_random_forest_model.pkl` | Best model (from GridSearchCV optimization) |
| `mc.ipynb` | MLflow integration and model comparison notebook |
| `02_data_modelling.ipynb` | Data cleaning and feature engineering logic |
