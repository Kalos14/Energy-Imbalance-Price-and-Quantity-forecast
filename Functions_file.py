## Sirus:Function files

from pathlib import Path
import numpy as np
import pandas as pd
import time
from sklearn.calibration import cross_val_predict
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from xgboost import XGBRegressor, XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib





def build_all_features(
    price: pd.DataFrame,
    quantity: pd.DataFrame,
    lag_steps=(1, 2, 4, 8, 24, 96),           # Empirically selected via ACF/PACF analysis
    roll_windows=(4, 8, 16, 32, 64),          # Rolling windows: 15min to 4 hours
    long_share_windows=(8, 16, 48, 96),       # Regime memory: 2 to 12 hours (justified by quantity ACF)
) -> pd.DataFrame:
    """
    Build engineered features for multi-step imbalance price and regime forecasting.
    
    All features are constructed using only information available at forecast time (no lookahead).
    Features are designed to capture: short-term momentum (lags), local volatility (rolling stats),
    calendar seasonality, intraday patterns, regime persistence, and price-quantity interactions.

    Parameters
    ----------
    price : pd.DataFrame
        DataFrame with columns ["flow_date", "mtu", "reference_date", "value"].
        - value: imbalance price in EUR/MWh
        - reference_date: UTC timestamp (enforces strict time ordering)
    quantity : pd.DataFrame
        DataFrame with identical structure as price.
        - value: imbalance quantity in MWh (positive = long/surplus, negative = short/deficit)
    lag_steps : tuple of int, optional
        Lags (in MTUs, 15-minute intervals) for autoregressive features.
        Default: (1, 2, 4, 8, 12, 24) → captures 15 minutes to 6 hours of history.
        Rationale: ACF analysis shows lag-1 dominates (~0.65 for price, ~0.80 for quantity),
        with meaningful but decaying correlation up to lag-24. Lags beyond 24 contribute
        diminishing signal and risk overfitting; PACF confirms AR(1) dominance.
    roll_windows : tuple of int, optional
        Rolling window sizes (MTUs) for mean/std computation.
        Default: (4, 8, 16, 32, 64) → 1 hour to 4.3 hours.
        These complement lags by capturing local volatility and smoothed trends,
        reducing multicollinearity risk from stacking many lagged features.
    long_share_windows : tuple of int, optional
        Windows (MTUs) for computing historical proportion of long regimes.
        Default: (8, 16, 32, 48) → 2 to 12 hours.
        Quantity ACF shows strong persistence; these capture regime persistence effects.

    Returns
    -------
    pd.DataFrame
        Feature matrix sorted by reference_date with ~80–100 features:
        - Base series: price, quantity, is_long (binary regime indicator)
        - Lagged features: price, quantity, is_long at selected lags
        - Rolling statistics: mean and std for price and quantity
        - Calendar features: day-of-week, weekend flag, month, hour_of_day, peak hours
        - Intraday seasonality: sine/cosine encoding (respects 24-hour cyclicity)
        - Regime memory: long_share (historical proportion long), long_streak, short_streak
        - Momentum: 1-step and multi-step price/quantity changes
        - Interactions: price × quantity decomposed by regime (long vs. short)
        
    Notes
    -----
    - All rolling operations use min_periods=1 to avoid NaN at series start, but rows with
      insufficient history should be dropped by caller before modeling (see data validation cell).
    - No external data (weather, load forecasts, day-ahead prices) is used, per project constraints.
    - Time ordering (reference_date) is strictly maintained; no sorting by other columns.
    """
    
    # =========================================================================
    # 1) MERGE BASE SERIES & DEFINE REGIME
    # =========================================================================
    # Rename 'value' columns to descriptive names and merge on time identifiers
    price_df = price.rename(columns={"value": "price"})
    qty_df = quantity.rename(columns={"value": "quantity"})

    df = (
        price_df.merge(
            qty_df,
            on=["flow_date", "mtu", "reference_date"],
            how="inner",
            validate="one_to_one",
        )
        .sort_values("reference_date")
        .reset_index(drop=True)
    )

    # Binary regime: 1 = long (surplus), 0 = short (deficit)
    # This is the target for probability forecasts (Task 2)
    df["is_long"] = (df["quantity"] > 0).astype(int)

    # Ensure flow_date is datetime (required for calendar features)
    if not np.issubdtype(df["flow_date"].dtype, np.datetime64):
        df["flow_date"] = pd.to_datetime(df["flow_date"])

    

    # =========================================================================
    # 2) AUTOREGRESSIVE FEATURES (Lags)
    # =========================================================================
    # Lag features capture short-term persistence. ACF analysis shows:
    # - Price lag-1 ACF ≈ 0.65 (strong immediate dependence)
    # - Quantity lag-1 ACF ≈ 0.80 (very strong; imbalances persist longer)
    # - PACF flattens after lag-1; subsequent lags refine but risk multicollinearity
    
    for l in lag_steps:
        df[f"price_lag_{l}"] = df["price"].shift(l)
        df[f"quantity_lag_{l}"] = df["quantity"].shift(l)
        df[f"is_long_lag_{l}"] = df["is_long"].shift(l)

    # =========================================================================
    # 3) ROLLING STATISTICS
    # =========================================================================
    # Rolling mean/std capture local trends and volatility without multicollinearity
    # issues inherent to many lagged features. Window sizes chosen to complement lags:
    # - 4 MTUs (1 hour): very local volatility
    # - 8 MTUs (2 hours): short-term trend
    # - 16 MTUs (4 hours): medium-term smoothing
    # - 32 MTUs (8 hours): longer-term baseline
    # - 64 MTUs (4.3 hours): captures half-day trends
    
    for w in roll_windows:
        df[f"price_roll_mean_{w}"] = df["price"].rolling(w, min_periods=1).mean()
        df[f"price_roll_std_{w}"] = df["price"].rolling(w, min_periods=1).std()

        df[f"quantity_roll_mean_{w}"] = df["quantity"].rolling(w, min_periods=1).mean()
        df[f"quantity_roll_std_{w}"] = df["quantity"].rolling(w, min_periods=1).std()

    # =========================================================================
    # 4) CALENDAR & TEMPORAL FEATURES
    # =========================================================================
    # No external data used; all features derived from timestamps and MTU index
    
    # Day-of-week and weekend flag
    df["dow"] = df["flow_date"].dt.weekday          # 0=Monday, 6=Sunday
    df["is_weekend"] = (df["dow"] >= 5).astype(int) # 1 if Sat/Sun
    df["month"] = df["flow_date"].dt.month           # 1-12

    # Hour of day: MTU ranges 1-96 per day; 4 MTUs = 1 hour
    df["hour_of_day"] = (df["mtu"] // 4).astype(int)  # 0-23

    # Peak hours: 08:00–20:00 (typical high-demand window in EU grids)
    # Imbalance prices often spike during peak demand and evening ramps
    df["is_peak_hour"] = ((df["hour_of_day"] >= 8) & (df["hour_of_day"] < 20)).astype(int)

    # =========================================================================
    # 5) INTRADAY SEASONALITY (Cyclical Encoding)
    # =========================================================================
    # Sine/cosine encoding respects the circular nature of time-of-day:
    # - Avoids artificial boundary discontinuity between MTU 95 (23:45) and MTU 0 (00:00)
    # - Allows linear models to learn smooth, periodic patterns
    # - Neural networks benefit from this representation as well
    
    df["mtu_sin"] = np.sin(2 * np.pi * df["mtu"] / 96.0)
    df["mtu_cos"] = np.cos(2 * np.pi * df["mtu"] / 96.0)

    # =========================================================================
    # 6) SEASONAL FEATURES
    # =========================================================================
    # Seasons affect generation mix (e.g., solar peaks in summer) and demand patterns
    
    month = df["month"]
    season = pd.Series(index=df.index, dtype="object")

    season[(month == 12) | (month <= 2)] = "winter"   # Dec, Jan, Feb
    season[(month >= 3) & (month <= 5)] = "spring"    # Mar, Apr, May
    season[(month >= 6) & (month <= 8)] = "summer"    # Jun, Jul, Aug
    season[(month >= 9) & (month <= 11)] = "autumn"   # Sep, Oct, Nov

    df["season"] = season.astype("category")

    # =========================================================================
    # 7) DAYLIGHT INDICATOR (Heuristic Proxy for Solar Generation)
    # =========================================================================
    # Note: This is NOT actual weather data.
    # Instead, it's a calendar-based heuristic that correlates with typical solar
    # generation windows in Central Europe. Trees/ensemble models will learn that
    # morning/midday imbalances during daylight hours often signal solar generation.
    # This heuristic is validated empirically by observing correlations in data.
    
    df["is_sun"] = 0

    # Summer: longer days, roughly 06:00–21:00
    mask_summer = df["season"] == "summer"
    df.loc[mask_summer & (df["hour_of_day"] >= 6) & (df["hour_of_day"] < 21), "is_sun"] = 1

    # Winter: shorter days, roughly 08:00–17:00
    mask_winter = df["season"] == "winter"
    df.loc[mask_winter & (df["hour_of_day"] >= 8) & (df["hour_of_day"] < 17), "is_sun"] = 1

    # Spring/Autumn: intermediate, roughly 07:00–20:00
    mask_spring_autumn = df["season"].isin(["spring", "autumn"])
    df.loc[
        mask_spring_autumn
        & (df["hour_of_day"] >= 7)
        & (df["hour_of_day"] < 20),
        "is_sun"
    ] = 1

    df["is_sun"] = df["is_sun"].astype(int)

    # =========================================================================
    # 8) GENERATION RAMP FEATURES
    # =========================================================================
    # Morning ramp (typically 07:00–09:00): Solar generation increases rapidly
    # Evening ramp (typically 18:00–20:00): Solar generation drops; demand peaks
    # These features help capture transition periods where imbalances are volatile
    
    df["is_morning_ramp"] = ((df["hour_of_day"] >= 7) & (df["hour_of_day"] < 9)).astype(int)
    df["is_evening_ramp"] = ((df["hour_of_day"] >= 18) & (df["hour_of_day"] < 20)).astype(int)

    # =========================================================================
    # 9) MOMENTUM FEATURES
    # =========================================================================
    # Capture first and higher-order differences to detect trend changes
    
    # 1-step quantity change: signals regime transitions (long → short or vice versa)
    df["dq_1"] = df["quantity"] - df["quantity_lag_1"]
    df["abs_dq_1"] = df["dq_1"].abs()  # Absolute change; magnitude matters regardless of sign

    # Multi-step price changes (1 hour and 24 hours back)
    # Detects momentum and mean reversion patterns
    for lag in (4, 96):
        col = f"price_lag_{lag}"
        if col in df.columns:
            df[f"dp_{lag}"] = df["price"] - df[col]

    # =========================================================================
    # 10) REGIME MEMORY FEATURES
    # =========================================================================
    # Capture persistence of long/short regimes. Quantity ACF shows strong daily
    # cycles (peaks every ~96 lags), meaning regimes repeat with daily frequency.
    
    # Historical proportion of long regime over sliding windows
    # Shift by 1 to ensure no lookahead: use only *past* regime values to predict future
    for window in long_share_windows:
        df[f"long_share_{window}"] = (
            df["is_long"].shift(1)
            .rolling(window, min_periods=1)
            .mean()
        )

    # Streak lengths: consecutive steps of being long or short
    # Long streaks → high probability next step also long; useful for regime persistence
    long_streak = np.zeros(len(df), dtype=int)
    short_streak = np.zeros(len(df), dtype=int)

    prev_long = 0
    prev_short = 0

    for i, val in enumerate(df["is_long"].values):
        if val == 1:
            prev_long += 1
            prev_short = 0
        else:
            prev_short += 1
            prev_long = 0
        long_streak[i] = prev_long
        short_streak[i] = prev_short

    df["long_streak"] = long_streak
    df["short_streak"] = short_streak

    # =========================================================================
    # 11) PRICE–QUANTITY INTERACTION FEATURES
    # =========================================================================
    # Decompose quantity by sign and interact with price
    # Rationale: Price response may differ sharply between surplus and deficit regimes
    
    # Decompose into positive (long) and negative (short) components
    df["q_pos"] = df["quantity"].clip(lower=0.0)    # Long quantity (surplus)
    df["q_neg"] = (-df["quantity"]).clip(lower=0.0)  # Short quantity magnitude (deficit)

    # Interactions: price response conditional on regime
    df["price_x_q"] = df["price"] * df["quantity"]              # Overall interaction
    df["price_x_q_pos"] = df["price"] * df["q_pos"]             # Price when long
    df["price_x_q_neg"] = df["price"] * df["q_neg"]             # Price when short

    # =========================================================================
    # RETURN
    # =========================================================================
    return df





def train_regression_models(
    df: pd.DataFrame,
    horizons=range(1, 25),
    save_dir: str | None = None,
    reg_type: str = "linear",  # "linear", "ridge", or "lasso"
    reg_alpha: float = 1.0,    # Optimal alpha (from separate CV func)
    use_residuals: bool = False,  # Regress residuals if True (add current_price at predict)
) -> dict:
    """
    Two-stage training for imbalance forecasting (linear baseline).


    Stage 1 (per horizon): Regression for quantity at horizon h.


    Stage 2 (per horizon): Use quantity_hat as feature for:
        - Conditional price (long/short regimes) → residuals if use_residuals
        - Prob(long regime) via logistic.


    Parameters
    ----------
    df : pd.DataFrame
        Feature DF from `build_all_features`; cols: price/quantity/is_long/ref_date + numerics.
    horizons : iterable of int
        MTU-ahead horizons (e.g., range(1,25)).
    save_dir : str or None
        Save models dict (joblib).
    reg_type : {"linear", "ridge", "lasso"}
        Regressor type.
    reg_alpha : float
        Ridge/Lasso strength (tune via CV).
    use_residuals : bool
        If True: regress r_long/short = P_target - current_price (stationary).
        Predict: Phat = current_price[-1] + r_hat.


    Returns
    -------
    dict:
        {"feature_cols": list, "horizons": list, "quantity": {h: model},
         "price_long": {h: model}, "price_short": {h: model},
         "prob_long": {h: model}, "reg_type": str, "reg_alpha": float,
         "uses_quantity_hat": True, "uses_price_residual": bool}
    """
    # Sort + select features (no mutate)
    assert df["reference_date"].is_monotonic_increasing, "Data not chrono-sorted!"

    exclude_cols = {"flow_date", "mtu", "reference_date"}
    numeric_cols = df.select_dtypes(include=["number"]).columns
    feature_cols = [c for c in numeric_cols if c not in exclude_cols]

    models_price_long = {}
    models_price_short = {}
    models_quantity = {}
    models_prob_long = {}

    def make_pipeline() -> Pipeline:
        """Scaler + regressor (consistent/CV-safe)."""
        reg = {"linear": LinearRegression(),
               "ridge": Ridge(alpha=reg_alpha),
               "lasso": Lasso(alpha=reg_alpha)}[reg_type]
        return Pipeline([("scaler", StandardScaler()), ("reg", reg)])

    for h in horizons:
        # Non-mutating targets
        price_targets = df["price"].shift(-h)
        quantity_targets = df["quantity"].shift(-h)
        regime_targets = df["is_long"].shift(-h)
        current_price = df["price"]  # For residuals

        cols_for_valid = feature_cols + ["price", "quantity", "is_long"]
        valid_mask = df[cols_for_valid].notna().all(axis=1) & price_targets.notna()
        tmp = df.loc[valid_mask].copy()

        if tmp.empty:
            raise ValueError(f"No valid samples for horizon {h}.")

        X_base = tmp[feature_cols].to_numpy()
        y_price = price_targets.loc[tmp.index].to_numpy()
        y_quantity = quantity_targets.loc[tmp.index].to_numpy()
        y_regime = regime_targets.loc[tmp.index].to_numpy().astype(int)
        price_current = current_price.loc[tmp.index].to_numpy()

        # Stage 1: quantity
        model_quantity = make_pipeline()
        model_quantity.fit(X_base, y_quantity)
        models_quantity[h] = model_quantity
        #This prevents the Price model from learning to trust a "perfect" quantity signal
        if len(X_base) > 50: # Only if enough data
            q_hat_train = cross_val_predict(model_quantity, X_base, y_quantity, cv=5).reshape(-1, 1)
        else:
            q_hat_train = model_quantity.predict(X_base).reshape(-1, 1)
        X_enriched = np.hstack([X_base, q_hat_train])

        # Stage 2: conditional price (residuals opt)
        mask_long = y_regime == 1
        mask_short = y_regime == 0

        X_long = X_enriched[mask_long]
        X_short = X_enriched[mask_short]

        if use_residuals:
            y_long = y_price[mask_long] - price_current[mask_long]
            y_short = y_price[mask_short] - price_current[mask_short]
        else:
            y_long = y_price[mask_long]
            y_short = y_price[mask_short]

        if len(y_long) == 0 or len(y_short) == 0:
            raise ValueError(f"Regime imbalance at h={h}: long={len(y_long)}, short={len(y_short)}.")

        model_long = make_pipeline()
        model_short = make_pipeline()
        model_long.fit(X_long, y_long)
        model_short.fit(X_short, y_short)

        models_price_long[h] = model_long
        models_price_short[h] = model_short

        # Stage 2: prob(long)
        unique_classes = np.unique(y_regime)
        if len(unique_classes) > 1:
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("logreg", LogisticRegression(max_iter=200, solver="lbfgs", penalty="l2"))
            ])
            clf.fit(X_enriched, y_regime)
        else:
            clf = DummyClassifier(strategy="constant", constant=int(unique_classes[0]))
            clf.fit(X_enriched, y_regime)

        models_prob_long[h] = clf

    models = {
        "feature_cols": feature_cols,
        "horizons": list(horizons),
        "quantity": models_quantity,
        "price_long": models_price_long,
        "price_short": models_price_short,
        "prob_long": models_prob_long,
        "reg_type": reg_type,
        "reg_alpha": reg_alpha,
        "uses_quantity_hat": True,
        "uses_price_residual": use_residuals,
    }

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(models, Path(save_dir) / "models_two_stage_linear.joblib")

    return models

from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, log_loss, brier_score_loss
import warnings

def evaluate_models_from_features(
    df: pd.DataFrame, models: dict, horizons=None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    assert df["reference_date"].is_monotonic_increasing, "Data not chrono-sorted!"

    feature_cols = models["feature_cols"]
    if horizons is None:
        horizons = models["horizons"]
    use_residuals = models.get("uses_price_residual", False)

    price_results, prob_results = [], []

    for h in horizons:
        # Non-mutating targets
        price_targets = df["price"].shift(-h)
        quantity_targets = df["quantity"].shift(-h)  # Unused but for mask
        regime_targets = df["is_long"].shift(-h)
        current_price = df["price"]  # For residuals/reconstruct

        valid_mask = (df[feature_cols].notna().all(1) &
                      price_targets.notna() & regime_targets.notna())
        tmp_idx = valid_mask[valid_mask].index  # Indices for targets
        if len(tmp_idx) < 100:
            warnings.warn(f"Low samples h={h}: {len(tmp_idx)}")

        X_base = df.loc[tmp_idx, feature_cols].to_numpy()
        y_price = price_targets.loc[tmp_idx].to_numpy()
        y_regime = regime_targets.loc[tmp_idx].to_numpy().astype(int)
        price_current = current_price.loc[tmp_idx].to_numpy()

        # Quantity_hat
        qty_model = models["quantity"][h]
        q_hat = qty_model.predict(X_base).reshape(-1, 1)
        X_enriched = np.hstack([X_base, q_hat])

        # Price preds (residual-aware)
        P_long_hat = models["price_long"][h].predict(X_enriched)
        P_short_hat = models["price_short"][h].predict(X_enriched)
        if use_residuals:
            P_long_hat += price_current  # Reconstruct
            P_short_hat += price_current
        P_hat = np.where(y_regime == 1, P_long_hat, P_short_hat)

        # Price metrics
        mae = mean_absolute_error(y_price, P_hat)
        rmse = np.sqrt(mean_squared_error(y_price, P_hat))
        price_results.append({"horizon": h, "mae": mae, "rmse": rmse, "n_samples": len(tmp_idx)})

        # Prob metrics (unchanged)
        clf = models["prob_long"][h]
        pi_hat = clf.predict_proba(X_enriched)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_enriched).astype(float)
        unique_classes = np.unique(y_regime)
        auc = roc_auc_score(y_regime, pi_hat) if len(unique_classes) > 1 else np.nan
        pi_hat_clipped = np.clip(pi_hat, 1e-6, 1 - 1e-6)
        ll = log_loss(y_regime, pi_hat_clipped)
        brier = brier_score_loss(y_regime, pi_hat_clipped)
        prob_results.append({"horizon": h, "auc": auc, "log_loss": ll, "brier": brier, "n_samples": len(tmp_idx)})

    return (pd.DataFrame(price_results).sort_values("horizon"),
            pd.DataFrame(prob_results).sort_values("horizon"))


def predict_24_steps_from_features(df_recent: pd.DataFrame, models: dict) -> pd.DataFrame:

    assert df_recent["reference_date"].is_monotonic_increasing, "Data not chrono-sorted!"

    if len(df_recent) == 0:
        raise ValueError("Empty df_recent.")
    
    last_row = df_recent.iloc[-1]
    feature_cols = models["feature_cols"]
    
    # Feats only (no mutate/NaN crash)
    if not last_row[feature_cols].notna().all():
        warnings.warn("NaNs in feats; filling 0.")
        X_last_base = last_row[feature_cols].fillna(0).to_numpy().reshape(1, -1)
    else:
        X_last_base = last_row[feature_cols].to_numpy().reshape(1, -1)
    
    price_last = float(last_row["price"])  # Residual base
    horizons = models["horizons"]
    use_residuals = models.get("uses_price_residual", False)
    use_qhat = models.get("uses_quantity_hat", True)

    h_list, q_hat_list = [], []
    P_long_list, P_short_list, pi_list = [], [], []

    for h in horizons:
        if use_qhat:
            q_hat_h = float(models["quantity"][h].predict(X_last_base)[0])
            q_hat_arr = np.array([[q_hat_h]])
            X_enrich = np.hstack([X_last_base, q_hat_arr])
        else:
            q_hat_h = np.nan
            X_enrich = X_last_base

        # Prices (resid-aware)
        P_long_hat = float(models["price_long"][h].predict(X_enrich)[0])
        P_short_hat = float(models["price_short"][h].predict(X_enrich)[0])
        if use_residuals:
            P_long_hat += price_last
            P_short_hat += price_last
        
        # Pi
        clf = models["prob_long"][h]
        pi_hat = (float(clf.predict_proba(X_enrich)[0, 1]) 
                  if hasattr(clf, "predict_proba") else float(clf.predict(X_enrich)[0]))
        
        h_list.append(h)
        q_hat_list.append(q_hat_h)
        P_long_list.append(P_long_hat)
        P_short_list.append(P_short_hat)
        pi_list.append(pi_hat)

    return pd.DataFrame({
        "horizon": h_list, "quantity_hat": q_hat_list,
        "P_long_hat": P_long_list, "P_short_hat": P_short_list, "pi_long_hat": pi_list
    })


def reliability_curve(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10, equal_count: bool = False) -> pd.DataFrame:
    """Binned calibration + ECE."""
    y_true, y_prob = np.asarray(y_true), np.clip(np.asarray(y_prob), 1e-6, 1 - 1e-6)
    if equal_count:
        bin_indices = pd.qcut(y_prob, n_bins, labels=False, duplicates="drop")
    else:
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    
    data = []
    for b in range(n_bins):
        mask = bin_indices == b
        if not mask.any(): continue
        data.append({
            "bin": b, "prob_mean": y_prob[mask].mean(),
            "event_rate": y_true[mask].mean(), "count": mask.sum()
        })
    df = pd.DataFrame(data)
    df["ece_contrib"] = np.abs(df["prob_mean"] - df["event_rate"]) * df["count"] / len(y_true)
    df["ece"] = df["ece_contrib"].sum()
    return df

def regression_results(price_metrics: pd.DataFrame, prob_metrics: pd.DataFrame,
                       forecasts: pd.DataFrame, save_dir: str | None = None) -> dict:
    """Tables/figs summary; saves PNGs."""
    buckets = [("1-4", slice(1,5)), ("5-12", slice(5,13)), ("13-24", slice(13,25))]
    
    # Price buckets
    rows = []
    for name, sl in buckets:
        mae_mean = price_metrics["mae"].iloc[sl].mean()
        rmse_mean = price_metrics["rmse"].iloc[sl].mean()
        rows.append({"horizon_range": name, "mae_mean": mae_mean, "rmse_mean": rmse_mean})
    price_buckets = pd.DataFrame(rows)
    
    # Prob buckets (+ Brier rel from notebook ~0.04 good)
    prob_buckets = (pd.DataFrame([{name: prob_metrics["auc"].iloc[h].mean(),
                                   "brier_mean": prob_metrics["brier"].iloc[h].mean(),
                                   "log_loss_mean": prob_metrics["log_loss"].iloc[h].mean()}
                                  for name, h in buckets]))
    prob_buckets.insert(0, "horizon_range", [name for name, _ in buckets])
    
    # E[price]; plots
    forecasts["E_price_hat"] = forecasts["pi_long_hat"] * forecasts["P_long_hat"] + \
                               (1 - forecasts["pi_long_hat"]) * forecasts["P_short_hat"]
    
    import matplotlib.pyplot as plt
    figs = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Price errors
    axes[0,0].plot(price_metrics["horizon"], price_metrics["rmse"], "o-", label="RMSE")
    axes[0,0].plot(price_metrics["horizon"], price_metrics["mae"], "x-", label="MAE")
    axes[0,0].set(xlabel="Horizon", ylabel="Error", title="Price Errors")
    axes[0,0].legend(); axes[0,0].grid(True)
    
    # AUC
    axes[0,1].plot(prob_metrics["horizon"], prob_metrics["auc"], "o-")
    axes[0,1].set(xlabel="Horizon", ylabel="AUC", title="Regime Discrimination")
    axes[0,1].grid(True)
    
    # Forecasts
    axes[1,0].plot(forecasts["horizon"], forecasts["P_long_hat"], label="P_long")
    axes[1,0].plot(forecasts["horizon"], forecasts["P_short_hat"], label="P_short")
    axes[1,0].plot(forecasts["horizon"], forecasts["E_price_hat"], "--", label="E[price]")
    axes[1,0].set(xlabel="Horizon", ylabel="EUR/MWh", title="Forecasts")
    axes[1,0].legend(); axes[1,0].grid(True)
    axes[1,1].plot(prob_metrics["horizon"], prob_metrics["brier"], "s-", color="purple")
    axes[1,1].set(xlabel="Horizon", ylabel="Brier Score", title="Prob. Calibration (Lower is Better)")
    axes[1,1].grid(True)
    
    # Usage: reliability = reliability_curve(y_regime_val, pi_val); plot it
    figs["summary"] = fig
    
    if save_dir:
        Path(save_dir).mkdir(exist_ok=True)
        fig.savefig(f"{save_dir}/results.png", dpi=150, bbox_inches="tight")
    
    plt.close(fig)
    return {"price_buckets": price_buckets, "prob_buckets": prob_buckets,
            "forecasts": forecasts, "figs": figs}


from sklearn.model_selection import TimeSeriesSplit
from joblib import Parallel, delayed
import warnings

def regression_cv(df: pd.DataFrame, alphas, r_type: str = "ridge",
                  horizons=range(1, 13),  # Short: Fast/prod
                  n_splits: int = 3, use_residuals: bool = True, n_jobs: int = -1) -> tuple[float, pd.DataFrame, pd.DataFrame]:

    assert df["reference_date"].is_monotonic_increasing, "Data not chrono-sorted!"

    tscv = TimeSeriesSplit(n_splits=n_splits, gap=24)  # 24 MTU gap to avoid leakage
    
    def cv_fold(alpha, fold_id, train_idx, val_idx):
        try:
            models = train_regression_models(df.iloc[train_idx], horizons=horizons,
                                             reg_type=r_type, reg_alpha=alpha,
                                             use_residuals=use_residuals)
            price_metrics, _ = evaluate_models_from_features(df.iloc[val_idx], models, horizons)
            if price_metrics.empty: return None
            return {"alpha": alpha, "fold": fold_id, "mean_rmse": price_metrics["rmse"].mean(),
                    "n_horizons": len(price_metrics)}
        except Exception as e:
            warnings.warn(f"Fold {fold_id}/α{alpha}: {e}")
            return None
    
    # Parallel: ~10x faster
    all_records = Parallel(n_jobs=n_jobs)(delayed(cv_fold)(alpha, fold_id, train_idx, val_idx)
                                         for alpha in alphas
                                         for fold_id, (train_idx, val_idx) in enumerate(tscv.split(df)))
    records = [r for r in all_records if r is not None]
    
    if not records:
        raise ValueError("No CV results.")
    
    cv_results = pd.DataFrame(records)
    summary = cv_results.groupby("alpha")["mean_rmse"].mean().sort_values().reset_index()
    best_alpha = float(summary.iloc[0]["alpha"])
    return best_alpha, cv_results, summary


import pandas as pd
import numpy as np
import warnings
import joblib
from pathlib import Path
from xgboost import XGBRegressor, XGBClassifier
from sklearn.dummy import DummyClassifier # FIXED import
from sklearn.model_selection import cross_val_predict, TimeSeriesSplit, ParameterGrid
from joblib import Parallel, delayed

def train_xgb_two_stage_residual_models(
    df: pd.DataFrame,
    horizons=range(1, 25),
    save_dir: str | None = None,
    xgb_reg_params: dict | None = None,
    xgb_clf_params: dict | None = None,
    use_residuals: bool = True,
) -> dict:

    assert pd.api.types.is_datetime64_any_dtype(df["reference_date"]), "Use datetime."
    assert df["reference_date"].is_monotonic_increasing, "Sort upstream!"
    
    # Default Params (Hist is faster)
    xgb_reg_params = xgb_reg_params or {
        "n_estimators": 100, "max_depth": 5, "learning_rate": 0.1,
        "objective": "reg:squarederror", "tree_method": "hist", "n_jobs": 1 
    }
    xgb_clf_params = xgb_clf_params or {
        "n_estimators": 100, "max_depth": 4, "learning_rate": 0.1,
        "objective": "binary:logistic", "eval_metric": "logloss", 
        "tree_method": "hist", "n_jobs": 1
    }
    
    exclude_cols = {"flow_date", "mtu", "reference_date"}
    feature_cols = [c for c in df.select_dtypes("number").columns if c not in exclude_cols]
    
    models = {
        "feature_cols": feature_cols, "horizons": list(horizons),
        "quantity": {}, "price_long": {}, "price_short": {}, "prob_long": {},
        "model_type": "xgboost_two_stage", "uses_quantity_hat": True,
        "uses_price_residual": use_residuals,
        "xgb_reg_params": xgb_reg_params, "xgb_clf_params": xgb_clf_params
    }
    
    # Dynamic keys based on residual usage
    price_long_key = "price_long_resid" if use_residuals else "price_long"
    price_short_key = "price_short_resid" if use_residuals else "price_short"
    
    # Prepare dictionaries in case they weren't init
    if price_long_key not in models: models[price_long_key] = {}
    if price_short_key not in models: models[price_short_key] = {}
    
    for h in horizons:
        # 1. Prepare Targets
        price_targs = df["price"].shift(-h)
        qty_targs = df["quantity"].shift(-h)
        regime_targs = df["is_long"].shift(-h)
        price_curr = df["price"]
        
        # 2. Valid Data Mask
        valid_mask = df[feature_cols].notna().all(axis=1) & \
                     price_targs.notna() & regime_targs.notna()

        if not valid_mask.any():
            warnings.warn(f"h={h}: No valid data."); continue
        
        # 3. Slice Data (Using numpy for speed)
        idx = df.index[valid_mask]
        X_base = df.loc[idx, feature_cols].to_numpy()
        y_qty = qty_targs.loc[idx].to_numpy()
        y_price = price_targs.loc[idx].to_numpy()
        y_regime = regime_targs.loc[idx].to_numpy().astype(int)
        price_c = price_curr.loc[idx].to_numpy()
        
        # Calculate Target (Residual or Raw)
        y_price_tgt = (y_price - price_c) if use_residuals else y_price
        
        # --- STAGE 1: Quantity Model ---
        qty_m = XGBRegressor(**xgb_reg_params)
        qty_m.fit(X_base, y_qty)
        models["quantity"][h] = qty_m
        
        # CRITICAL FIX: Use CV for internal features to avoid leakage
        # If we just predicted X_base, XGBoost would have 0 error and 
        # Stage 2 would rely 100% on a "perfect" quantity that doesn't exist at test time.
        if len(X_base) > 100:
            q_hat_train = cross_val_predict(
                XGBRegressor(**xgb_reg_params), X_base, y_qty, cv=5, n_jobs=1
            ).reshape(-1, 1)
        else:
            q_hat_train = qty_m.predict(X_base).reshape(-1, 1)

        X_enrich = np.hstack([X_base, q_hat_train])
        
        # --- STAGE 2: Conditional Price Models ---
        mask_l = y_regime == 1
        mask_s = y_regime == 0
        
        if not (mask_l.any() and mask_s.any()):
            warnings.warn(f"h={h}: One regime is empty (Long:{mask_l.sum()}, Short:{mask_s.sum()}). Skipping.")
            continue
        
        long_m = XGBRegressor(**xgb_reg_params)
        long_m.fit(X_enrich[mask_l], y_price_tgt[mask_l])
        
        short_m = XGBRegressor(**xgb_reg_params)
        short_m.fit(X_enrich[mask_s], y_price_tgt[mask_s])
        
        models[price_long_key][h] = long_m
        models[price_short_key][h] = short_m
        
        # --- STAGE 2: Probability Model ---
        uniq = np.unique(y_regime)
        if len(uniq) > 1:
            clf = XGBClassifier(**xgb_clf_params)
            clf.fit(X_enrich, y_regime)
        else:
            # Fallback if only one class exists in this window
            clf = DummyClassifier(strategy="constant", constant=int(uniq[0]))
            clf.fit(X_enrich, y_regime)
            
        models["prob_long"][h] = clf
    
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        joblib.dump(models, Path(save_dir)/"models_xgb.joblib")
        
    return models


def xgb_cv(
    df_features: pd.DataFrame,
    param_grid: dict,
    horizons=range(1, 7),  # Keep short for speed (e.g., 1 to 6)
    n_splits: int = 3,
    use_residuals: bool = True,
    n_jobs: int = -1,
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """
    Time-Series CV for XGBoost hyperparameters.
    """
    assert df_features["reference_date"].is_monotonic_increasing, "Data not chrono-sorted!"
    
    # Gap=24 ensures Validation targets don't overlap with Train lags
    # Note: 'gap' requires sklearn >= 1.0
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=24) 
    
    param_list = list(ParameterGrid(param_grid))
    
    # Need to import evaluate function inside or ensure it's in scope
    # from model_utils import evaluate_models_from_features
    
    def cv_fold(params, fold_id, train_idx, val_idx):
        # Force n_jobs=1 inside fold to avoid exploding threads
        reg_params = {**params, "objective": "reg:squarederror", "tree_method": "hist", "n_jobs": 1}
        clf_params = {**params, "objective": "binary:logistic", "tree_method": "hist", "n_jobs": 1} # Simplify: share params
        
        try:
            models = train_xgb_two_stage_residual_models(
                df_features.iloc[train_idx], 
                horizons=horizons, 
                use_residuals=use_residuals,
                xgb_reg_params=reg_params,
                xgb_clf_params=clf_params
            )
            # Evaluate
            price_m, _ = evaluate_models_from_features(df_features.iloc[val_idx], models, horizons)
            
            if price_m.empty: return None
            
            return {
                "params": params, # Store actual dict, not string
                "config_str": str(params),
                "fold": fold_id, 
                "mean_rmse": price_m["rmse"].mean(),
                "n_horiz": len(price_m)
            }
        except Exception as e:
            warnings.warn(f"Fold {fold_id} failed: {e}")
            return None
    
    # Run Parallel
    records = Parallel(n_jobs=n_jobs)(delayed(cv_fold)(p, fid, t, v)
                                      for p in param_list
                                      for fid, (t, v) in enumerate(tscv.split(df_features)))
    
    records = [r for r in records if r is not None]
    if not records:
        raise ValueError("No CV results returned.")
    
    cv_df = pd.DataFrame(records)
    
    # Aggregating results
    # We turn the 'params' dict into a string just for grouping, 
    # then pick the first actual 'params' dict from the group.
    summary = cv_df.groupby("config_str").agg({
        "mean_rmse": "mean",
        "params": "first" # Grab the dictionary back
    }).sort_values("mean_rmse").reset_index()
    
    best_params = summary.iloc[0]["params"]
    
    return best_params, cv_df, summary