import pandas as pd
import numpy as np
import logging
import joblib
from pathlib import Path
import matplotlib.pyplot as plt

# --- IMPORTS FROM YOUR MODULES ---
# Save the functions from the previous steps into 'model_utils.py'
from Functions_file import *

# --- CONFIGURATION ---
DATA_DIR = Path("Datasets")  # Folder containing imb_price.csv and imb_quantity.csv
OUTPUT_DIR = Path("output")
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# --- CUSTOM IMPUTATION LOGIC (For Part 0 of Report) ---
def handle_missing_prices(df_price: pd.DataFrame, df_qty: pd.DataFrame) -> pd.DataFrame:
    """
    Fills missing rows in Price based on Quantity timestamps.
    Imputation Strategy: Conditional Mean (Month + MTU + Regime).
    """
    logger.info("Starting Price Imputation...")
    
    # 1. Standardize Dates
    df_price['reference_date'] = pd.to_datetime(df_price['reference_date'], utc=True)
    df_qty['reference_date'] = pd.to_datetime(df_qty['reference_date'], utc=True)
    
    # 2. Identify Missing Timestamps (Price vs Quantity)
    # We expect Quantity to have the full range
    full_index = df_qty[['reference_date', 'mtu', 'flow_date']].copy()
    
    # Merge Price onto the full Quantity index (Left Join)
    df_merged = full_index.merge(df_price, on=['reference_date', 'mtu', 'flow_date'], how='left')
    
    # Identify gaps
    missing_mask = df_merged['value'].isna()
    n_missing = missing_mask.sum()
    logger.info(f"Found {n_missing} missing price values (likely 1 day).")
    
    if n_missing == 0:
        return df_price

    # 3. Prepare Data for Imputation Calculation
    # We need the Quantity value for the missing rows to know the Regime (Long/Short)
    df_merged = df_merged.merge(df_qty[['reference_date', 'value']], 
                                on='reference_date', suffixes=('_price', '_qty'))
    
    df_merged['month'] = df_merged['reference_date'].dt.month
    df_merged['is_long'] = df_merged['value_qty'] > 0
    
    # 4. Build Lookup Table (Mean Price per Month, MTU, Regime)
    # Use only existing data
    lookup = df_merged.dropna(subset=['value_price']).groupby(
        ['month', 'mtu', 'is_long']
    )['value_price'].mean().reset_index()
    
    # 5. Fill NaNs
    # Merge the lookup table back to the main df
    df_imputed = df_merged.merge(lookup, on=['month', 'mtu', 'is_long'], how='left', suffixes=('', '_mean'))
    
    # Fill missing prices with the calculated mean
    df_imputed['value_price'] = df_imputed['value_price'].fillna(df_imputed['value_price_mean'])
    
    # 6. Cleanup to return original format
    df_final = df_imputed[['reference_date', 'flow_date', 'mtu', 'value_price']].rename(
        columns={'value_price': 'value'}
    ).sort_values('reference_date')
    
    logger.info("Imputation complete.")
    return df_final

# --- MAIN EXECUTION ---
def main():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # 1. Load Data
    logger.info("Loading Data...")
    try:
        df_price = pd.read_parquet(DATA_DIR / "imb_price.parquet")
        df_qty = pd.read_parquet(DATA_DIR / "imb_quantity.parquet")
    except FileNotFoundError:
        logger.error(f"Data files not found in {DATA_DIR}. Please check paths.")
        return

    # 2. Imputation (Fixing the 96 row gap)
    df_price_clean = handle_missing_prices(df_price, df_qty)
    
    # 3. Feature Engineering
    logger.info("Building Features (this may take a moment)...")
    # Note: Ensure build_all_features inside model_utils uses the cleaned DFs
    df_features = build_all_features(df_price_clean, df_qty)
    
    # Drop initial NaNs created by lags (first 24 hours usually)
    df_features = df_features.dropna().reset_index(drop=True)
    logger.info(f"Feature matrix shape: {df_features.shape}")

    # 4. Train/Test Split (Time-based)
    # We hold out the last 30 days for Final Evaluation
    TEST_DAYS = 30
    cutoff_idx = len(df_features) - (TEST_DAYS * 96)
    
    train_df = df_features.iloc[:cutoff_idx].copy()
    test_df = df_features.iloc[cutoff_idx:].copy()
    
    logger.info(f"Train set: {len(train_df)} rows. Test set: {len(test_df)} rows.")
    # =========================================================
    # PART 0: Linear Regression )
    # =========================================================
    logger.info("\n=== Starting Linear Regression Pipeline ===")
    lin_reg = train_regression_models(
        train_df,
        horizons=range(1, 25),
        save_dir=OUTPUT_DIR / "linear_models",
        reg_type="linear",
        reg_alpha=0.0,
        use_residuals=False,
    )

    logger.info("Evaluating Linear Regression on Test Set...")
    lin_price_metrics, lin_prob_metrics = evaluate_models_from_features(
        test_df, lin_reg, horizons=range(1, 25)
    )

    lin_forecast_example = predict_24_steps_from_features(test_df.iloc[:96*5], lin_reg)

    lin_results_summary = regression_results(
        lin_price_metrics, 
        lin_prob_metrics, 
        lin_forecast_example, # Placeholder for forecast dataframe (optional for summary plot)
        save_dir=OUTPUT_DIR / "linear_plots"
    )
    # Save raw metrics for the LaTeX report
    lin_price_metrics.to_csv(OUTPUT_DIR / "metrics_linear_price.csv", index=False)
    lin_prob_metrics.to_csv(OUTPUT_DIR / "metrics_linear_prob.csv", index=False)


    # =========================================================
    # PART 1: Ridge Regression
    # =========================================================
    logger.info("\n=== Starting Ridge Regression Pipeline ===")

    # 5. Cross-Validation (Hyperparameter Tuning)
    logger.info("Running Cross-Validation for Ridge Alpha...")
    alphas_to_test = [1.0, 10.0, 100.0, 1000.0]
    
    # Using the CV function (Ensure regression_cv in model_utils has the gap=24 fix if possible)
    best_alpha, cv_results, summary = regression_cv(
        train_df, 
        alphas=alphas_to_test, 
        r_type="ridge", 
        horizons=range(1, 13), # Check first 12 hours for speed
        n_splits=3,
        use_residuals=True,
        n_jobs=1
    )
    logger.info(f"Best Alpha found: {best_alpha}")
    print(summary)

    # 6. Final Model Training
    logger.info("Training Final Models (Horizons 1-24)...")
    models = train_regression_models(
        train_df,
        horizons=range(1, 25),
        save_dir=OUTPUT_DIR / "models",
        reg_type="ridge",
        reg_alpha=best_alpha,
        use_residuals=True,
    )

    # 7. Evaluation on Test Set
    logger.info("Evaluating on Test Set...")
    price_metrics, prob_metrics = evaluate_models_from_features(
        test_df, models, horizons=range(1, 25)
    )
    
    # Generate Forecasts for visual inspection (First 5 days of test)
    subset_test = test_df.iloc[:96*5]
    forecast_example = predict_24_steps_from_features(subset_test, models) # Note: this func predicts from LAST row. 
    # For full evaluation plots, we rely on evaluate_models returning metrics.
    # To get a plot of predictions vs Actuals, we need to run prediction over the test set loop.
    # For the report, we use the `regression_results` function which handles aggregation.
    
    results_summary = regression_results(
        price_metrics, 
        prob_metrics, 
        forecast_example, # forecast_df placeholder if not fully generated
        save_dir=OUTPUT_DIR / "Regression_plots"
    )
    
    # Save Metrics to CSV for LaTeX report
    price_metrics.to_csv(OUTPUT_DIR / "metrics_price_regression.csv", index=False)
    prob_metrics.to_csv(OUTPUT_DIR / "metrics_prob_regression.csv", index=False)

    # =========================================================
    # PART 2: XGBoost 
    # =========================================================
    logger.info("\n=== Starting XGBoost Pipeline ===")


    # 1. XGB Cross-Validation
    logger.info("Running XGBoost Cross-Validation...")
    
    # Hyperparameter Grid
    # Kept lightweight for demonstration speed. 
    # In production, expand n_estimators and max_depth ranges.
    # xgb_param_grid = {
    #     "learning_rate": [0.05, 0.1],
    #     "max_depth": [4, 6],
    #     "n_estimators": [200, 300]
    # }
    
    # best_xgb_params, xgb_cv_results, xgb_summary = xgb_cv(
    #     train_df,
    #     param_grid=xgb_param_grid,
    #     horizons=range(1, 6), # CV on first 6 MTU to save time
    #     n_splits=3,
    #     use_residuals=True,    # XGB works better predicting the 'change' (residual) rather than absolute value
    #     n_jobs=1               
    # )

    logger.info("Configuring XGBoost with optimized parameters...")
    
    best_xgb_params = {
        "n_estimators": 200,       
        "learning_rate": 0.1,     
        "max_depth": 4,            
        "subsample": 0.8,          
        "colsample_bytree": 0.8,   
        "n_jobs": 1,               
        "tree_method": "hist",     
        "random_state": 42
    }

    logger.info(f"Best XGB Params: {best_xgb_params}")

    # 2. Final XGB Training
    logger.info("Training Final XGBoost Models (Horizons 1-24)...")
    
    # Prepare parameters for the final fit
    # We enforce single-threading (n_jobs=1) per model to avoid OOM, 
    # as we might train multiple horizons sequentially or in parallel depending on future implementation.

    
    xgb_models = train_xgb_two_stage_residual_models(
        train_df,
        horizons=range(1, 25),
        save_dir=OUTPUT_DIR / "models_xgb",
        xgb_reg_params=best_xgb_params, # Applying same params to Regressor and Classifier for simplicity
        xgb_clf_params=best_xgb_params,
        use_residuals=True
    )



    # 3. Evaluation on Test Set
    logger.info("Evaluating XGBoost on Test Set...")
    xgb_price_metrics, xgb_prob_metrics = evaluate_models_from_features(
        test_df, xgb_models, horizons=range(1, 25)
    )

    xgb_forecasts = generate_full_forecasts(test_df, xgb_models)
    xgb_forecasts.to_csv(OUTPUT_DIR / "forecasts_xgb.csv", index=False)
    
    rename_map = {
        "pred_price_long": "P_long_hat",
        "pred_price_short": "P_short_hat",
        "pred_prob_long": "pi_long_hat"
    }
    
    xgb_forecasts.rename(columns=rename_map, inplace=True)
    # 4. Save Results & Plots
    # Create a specific subdirectory for XGBoost plots to distinguish from Linear plots
    (OUTPUT_DIR / "xgb_plots").mkdir(exist_ok=True)
    
    xgb_results_summary = regression_results(
        xgb_price_metrics, 
        xgb_prob_metrics, 
        xgb_forecasts,
        save_dir=OUTPUT_DIR / "xgb_plots"
    )
    
    # Save raw metrics for the LaTeX report
    xgb_price_metrics.to_csv(OUTPUT_DIR / "metrics_xgb_price.csv", index=False)
    xgb_prob_metrics.to_csv(OUTPUT_DIR / "metrics_xgb_prob.csv", index=False)

# =========================================================
    # PART 3: CatBoost 
    # =========================================================
    logger.info("\n=== Starting CatBoost Pipeline ===")
    
    try:
        # Configuration for CatBoost
        logger.info("Configuring CatBoost (Categorical Support Enabled)...")
        final_cat_params = {
            "iterations": 300,        # CatBoost is fast, 500 is safe
            "depth": 4,
            "learning_rate": 0.1,
            "loss_function": "RMSE",
            "verbose": 0,
            "allow_writing_files": False,
            "thread_count": 1         # Low memory usage
        }
        
        # Training
        logger.info("Training CatBoost Models...")
        cat_models = train_catboost_two_stage_residual_models(
            train_df, horizons=range(1, 25), save_dir=OUTPUT_DIR / "models_catboost",
            cat_params=final_cat_params, use_residuals=True
        )
        
        # Evaluation
        logger.info("Evaluating CatBoost...")
        cat_price_metrics, cat_prob_metrics = evaluate_models_from_features(
            test_df, cat_models, horizons=range(1, 25)
        )
        
        # Forecast Generation
        logger.info("Generating full CatBoost forecasts...")
        cat_forecasts = generate_full_forecasts(test_df, cat_models)
        cat_forecasts.to_csv(OUTPUT_DIR / "forecasts_catboost.csv", index=False)
        
        # Save Results
        (OUTPUT_DIR / "cat_plots").mkdir(exist_ok=True)
        regression_results(cat_price_metrics, cat_prob_metrics, pd.DataFrame(), save_dir=OUTPUT_DIR / "cat_plots")
        cat_price_metrics.to_csv(OUTPUT_DIR / "metrics_cat_price.csv", index=False)
        cat_rmse = cat_price_metrics['rmse'].mean()
        
    except ImportError:
        logger.warning("CatBoost not installed. Skipping Part 3.")
        cat_rmse = np.nan
    except Exception as e:
        logger.error(f"CatBoost failed: {e}")
        cat_rmse = np.nan

    # =========================================================
    # FINAL LEADERBOARD
    # =========================================================
    lin_rmse = pd.read_csv(OUTPUT_DIR / "metrics_linear_price.csv")['rmse'].mean()
    ridge_rmse = pd.read_csv(OUTPUT_DIR / "metrics_price_regression.csv")['rmse'].mean()
    xgb_rmse = pd.read_csv(OUTPUT_DIR / "metrics_xgb_price.csv")['rmse'].mean() 
    cat_rmse = pd.read_csv(OUTPUT_DIR / "metrics_cat_price.csv")['rmse'].mean()
    logger.info("\n" + "="*40)
    logger.info("FINAL MODEL LEADERBOARD (RMSE)")
    logger.info("="*40)
    logger.info(f"0. Linear Baseline: {lin_rmse:.4f} EUR/MWh")
    logger.info(f"1. Ridge Regression: {ridge_rmse:.4f} EUR/MWh")
    logger.info(f"2. XGBoost:         {xgb_rmse:.4f} EUR/MWh")
    if not np.isnan(cat_rmse):
        logger.info(f"3. CatBoost:        {cat_rmse:.4f} EUR/MWh")
        
    logger.info("="*40)
    logger.info(f"Pipeline Finished. Check {OUTPUT_DIR} for plots and CSVs.")

if __name__ == "__main__":
    main()