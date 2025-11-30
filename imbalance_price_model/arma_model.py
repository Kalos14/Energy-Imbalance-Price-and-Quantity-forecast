"""
ARMA Model for Imbalance Price Forecasting.

This module implements an ARMA (AutoRegressive Moving Average) model
for forecasting energy imbalance prices with arcsinh transformation
for stationarity.
"""

import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss

from .transforms import arcsinh_transform, inverse_arcsinh_transform


class ImbalancePriceARMA:
    """
    ARMA model for energy imbalance price forecasting.
    
    This model uses an arcsinh(x/std)/std transformation to achieve
    stationarity before fitting an ARMA model. The transformation
    is particularly suited for imbalance prices which can have
    heavy tails and both positive and negative values.
    
    Parameters
    ----------
    p : int, default=1
        The order of the AR (autoregressive) component
    q : int, default=1
        The order of the MA (moving average) component
        
    Attributes
    ----------
    p : int
        AR order
    q : int
        MA order
    std_ : float
        Standard deviation used for transformation (fitted)
    model_ : ARIMA
        Fitted ARIMA model (ARMA is ARIMA with d=0)
    results_ : ARIMAResults
        Results from the fitted model
        
    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> # Generate sample imbalance price data
    >>> prices = np.cumsum(np.random.randn(200)) + 50
    >>> 
    >>> model = ImbalancePriceARMA(p=2, q=1)
    >>> model.fit(prices)
    >>> forecast = model.predict(steps=10)
    >>> print(f"Forecast shape: {forecast.shape}")
    """
    
    def __init__(self, p: int = 1, q: int = 1):
        """Initialize the ARMA model with specified orders."""
        if p < 0:
            raise ValueError("AR order p must be non-negative")
        if q < 0:
            raise ValueError("MA order q must be non-negative")
            
        self.p = p
        self.q = q
        self.std_ = None
        self.model_ = None
        self.results_ = None
        self._transformed_data = None
        
    def fit(
        self, 
        data: Union[np.ndarray, pd.Series],
        std: Optional[float] = None
    ) -> 'ImbalancePriceARMA':
        """
        Fit the ARMA model to imbalance price data.
        
        The data is first transformed using arcsinh(x/std)/std to
        achieve stationarity, then an ARMA(p,q) model is fitted.
        
        Parameters
        ----------
        data : array-like
            Imbalance price timeseries data
        std : float, optional
            Standard deviation for transformation. If None, computed from data.
            
        Returns
        -------
        self : ImbalancePriceARMA
            Fitted model instance
        """
        # Convert to numpy array
        if isinstance(data, pd.Series):
            data = data.values
        data = np.asarray(data, dtype=np.float64)
        
        # Check for NaN values
        if np.any(np.isnan(data)):
            raise ValueError("Data contains NaN values")
        
        # Apply arcsinh transformation for stationarity
        self._transformed_data, self.std_ = arcsinh_transform(data, std)
        
        # Fit ARMA model (ARIMA with d=0)
        # Using ARIMA with d=0 is equivalent to ARMA
        try:
            self.model_ = ARIMA(
                self._transformed_data, 
                order=(self.p, 0, self.q)
            )
            self.results_ = self.model_.fit()
        except Exception as e:
            raise RuntimeError(
                f"Failed to fit ARMA({self.p},{self.q}) model. "
                f"This may be due to convergence issues or invalid model orders. "
                f"Consider trying different p, q values or checking data quality. "
                f"Original error: {e}"
            ) from e
        
        return self
    
    def predict(
        self, 
        steps: int = 1,
        return_conf_int: bool = False,
        alpha: float = 0.05
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Forecast future imbalance prices.
        
        Parameters
        ----------
        steps : int, default=1
            Number of steps ahead to forecast
        return_conf_int : bool, default=False
            Whether to return confidence intervals
        alpha : float, default=0.05
            Significance level for confidence intervals
            
        Returns
        -------
        forecast : np.ndarray
            Point forecasts in original scale
        lower : np.ndarray (only if return_conf_int=True)
            Lower confidence interval bounds
        upper : np.ndarray (only if return_conf_int=True)
            Upper confidence interval bounds
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Get forecast in transformed space
        forecast_obj = self.results_.get_forecast(steps=steps)
        forecast_transformed = forecast_obj.predicted_mean
        
        # Inverse transform to get original scale
        forecast = inverse_arcsinh_transform(forecast_transformed, self.std_)
        
        if return_conf_int:
            conf_int = forecast_obj.conf_int(alpha=alpha)
            # Handle both DataFrame and ndarray return types
            if isinstance(conf_int, pd.DataFrame):
                lower = inverse_arcsinh_transform(conf_int.iloc[:, 0].values, self.std_)
                upper = inverse_arcsinh_transform(conf_int.iloc[:, 1].values, self.std_)
            else:
                lower = inverse_arcsinh_transform(conf_int[:, 0], self.std_)
                upper = inverse_arcsinh_transform(conf_int[:, 1], self.std_)
            return forecast, lower, upper
        
        return forecast
    
    def get_fitted_values(self) -> np.ndarray:
        """
        Get fitted values in original scale.
        
        Returns
        -------
        fitted : np.ndarray
            Fitted values from the model in original scale
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted first")
        
        fitted_transformed = self.results_.fittedvalues
        return inverse_arcsinh_transform(fitted_transformed, self.std_)
    
    def get_residuals(self, in_original_scale: bool = False) -> np.ndarray:
        """
        Get model residuals.
        
        Parameters
        ----------
        in_original_scale : bool, default=False
            If True, return residuals in original scale.
            If False, return residuals in transformed scale.
            
        Returns
        -------
        residuals : np.ndarray
            Model residuals
        """
        if self.results_ is None:
            raise ValueError("Model must be fitted first")
        
        if in_original_scale:
            original_data = inverse_arcsinh_transform(
                self._transformed_data, self.std_
            )
            fitted = self.get_fitted_values()
            return original_data - fitted
        else:
            return self.results_.resid
    
    def check_stationarity(
        self, 
        data: Optional[np.ndarray] = None,
        significance: float = 0.05
    ) -> dict:
        """
        Test stationarity of the (transformed) data.
        
        Uses both the Augmented Dickey-Fuller (ADF) test and 
        KPSS test for stationarity testing.
        
        Parameters
        ----------
        data : np.ndarray, optional
            Data to test. If None, uses transformed training data.
        significance : float, default=0.05
            Significance level for the tests
            
        Returns
        -------
        results : dict
            Dictionary containing test results with keys:
            - 'adf_statistic': ADF test statistic
            - 'adf_pvalue': ADF p-value
            - 'adf_is_stationary': Boolean indicating stationarity
            - 'kpss_statistic': KPSS test statistic
            - 'kpss_pvalue': KPSS p-value
            - 'kpss_is_stationary': Boolean indicating stationarity
        """
        if data is None:
            if self._transformed_data is None:
                raise ValueError("No data available. Fit the model first or provide data.")
            data = self._transformed_data
        
        # ADF test (null hypothesis: unit root exists, i.e., non-stationary)
        adf_result = adfuller(data, autolag='AIC')
        adf_stationary = adf_result[1] < significance
        
        # KPSS test (null hypothesis: stationary)
        # Using 'c' for constant (level stationarity)
        kpss_result = kpss(data, regression='c', nlags='auto')
        kpss_stationary = kpss_result[1] > significance
        
        return {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'adf_is_stationary': adf_stationary,
            'kpss_statistic': kpss_result[0],
            'kpss_pvalue': kpss_result[1],
            'kpss_is_stationary': kpss_stationary
        }
    
    def summary(self) -> str:
        """
        Get a summary of the fitted model.
        
        Returns
        -------
        summary : str
            String summary of the model
        """
        if self.results_ is None:
            return "Model not fitted yet"
        return str(self.results_.summary())
    
    @property
    def aic(self) -> float:
        """Akaike Information Criterion of the fitted model."""
        if self.results_ is None:
            raise ValueError("Model must be fitted first")
        return self.results_.aic
    
    @property
    def bic(self) -> float:
        """Bayesian Information Criterion of the fitted model."""
        if self.results_ is None:
            raise ValueError("Model must be fitted first")
        return self.results_.bic
