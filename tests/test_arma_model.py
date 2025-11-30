"""
Tests for the ARMA model.
"""

import numpy as np
import pandas as pd
import pytest

from imbalance_price_model.arma_model import ImbalancePriceARMA


def generate_sample_data(n=200, seed=42):
    """Generate sample imbalance price data for testing."""
    np.random.seed(seed)
    # Generate data that looks like imbalance prices
    # Random walk with some mean reversion
    data = np.zeros(n)
    data[0] = 50  # Starting price
    for i in range(1, n):
        data[i] = 0.9 * data[i-1] + 0.1 * 50 + np.random.randn() * 10
    return data


class TestImbalancePriceARMAInit:
    """Tests for ImbalancePriceARMA initialization."""
    
    def test_default_initialization(self):
        """Test default initialization with p=1, q=1."""
        model = ImbalancePriceARMA()
        
        assert model.p == 1
        assert model.q == 1
        assert model.std_ is None
        assert model.model_ is None
        assert model.results_ is None
    
    def test_custom_initialization(self):
        """Test initialization with custom p and q."""
        model = ImbalancePriceARMA(p=3, q=2)
        
        assert model.p == 3
        assert model.q == 2
    
    def test_invalid_p_raises_error(self):
        """Test that negative p raises ValueError."""
        with pytest.raises(ValueError, match="AR order p must be non-negative"):
            ImbalancePriceARMA(p=-1, q=1)
    
    def test_invalid_q_raises_error(self):
        """Test that negative q raises ValueError."""
        with pytest.raises(ValueError, match="MA order q must be non-negative"):
            ImbalancePriceARMA(p=1, q=-1)


class TestImbalancePriceARMAFit:
    """Tests for the fit method."""
    
    def test_fit_with_numpy_array(self):
        """Test fitting with numpy array."""
        data = generate_sample_data()
        model = ImbalancePriceARMA(p=1, q=1)
        
        result = model.fit(data)
        
        assert result is model  # Should return self
        assert model.std_ is not None
        assert model.std_ > 0
        assert model.results_ is not None
    
    def test_fit_with_pandas_series(self):
        """Test fitting with pandas Series."""
        data = pd.Series(generate_sample_data())
        model = ImbalancePriceARMA(p=1, q=1)
        
        model.fit(data)
        
        assert model.results_ is not None
    
    def test_fit_with_custom_std(self):
        """Test fitting with custom std value."""
        data = generate_sample_data()
        model = ImbalancePriceARMA(p=1, q=1)
        custom_std = 25.0
        
        model.fit(data, std=custom_std)
        
        assert model.std_ == custom_std
    
    def test_fit_with_nan_raises_error(self):
        """Test that data with NaN raises ValueError."""
        data = generate_sample_data()
        data[10] = np.nan
        model = ImbalancePriceARMA(p=1, q=1)
        
        with pytest.raises(ValueError, match="Data contains NaN values"):
            model.fit(data)
    
    def test_fit_different_orders(self):
        """Test fitting with different AR and MA orders."""
        data = generate_sample_data()
        
        for p, q in [(0, 1), (1, 0), (2, 1), (1, 2), (2, 2)]:
            model = ImbalancePriceARMA(p=p, q=q)
            model.fit(data)
            assert model.results_ is not None


class TestImbalancePriceARMAPredict:
    """Tests for the predict method."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model for testing."""
        data = generate_sample_data()
        model = ImbalancePriceARMA(p=1, q=1)
        model.fit(data)
        return model
    
    def test_predict_single_step(self, fitted_model):
        """Test single-step prediction."""
        forecast = fitted_model.predict(steps=1)
        
        assert isinstance(forecast, np.ndarray)
        assert len(forecast) == 1
    
    def test_predict_multiple_steps(self, fitted_model):
        """Test multi-step prediction."""
        steps = 10
        forecast = fitted_model.predict(steps=steps)
        
        assert len(forecast) == steps
    
    def test_predict_with_confidence_intervals(self, fitted_model):
        """Test prediction with confidence intervals."""
        forecast, lower, upper = fitted_model.predict(
            steps=5, return_conf_int=True
        )
        
        assert len(forecast) == 5
        assert len(lower) == 5
        assert len(upper) == 5
        # Lower should be less than upper
        assert np.all(lower <= upper)
    
    def test_predict_without_fit_raises_error(self):
        """Test that predicting without fitting raises error."""
        model = ImbalancePriceARMA(p=1, q=1)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            model.predict(steps=1)
    
    def test_predictions_are_reasonable(self, fitted_model):
        """Test that predictions are within reasonable range."""
        forecast = fitted_model.predict(steps=10)
        
        # Predictions should not be extreme (within a reasonable range for our test data)
        assert np.all(np.abs(forecast) < 1000)


class TestImbalancePriceARMAResiduals:
    """Tests for residual-related methods."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model for testing."""
        data = generate_sample_data()
        model = ImbalancePriceARMA(p=1, q=1)
        model.fit(data)
        return model
    
    def test_get_fitted_values(self, fitted_model):
        """Test getting fitted values."""
        fitted = fitted_model.get_fitted_values()
        
        assert isinstance(fitted, np.ndarray)
        assert len(fitted) == 200
    
    def test_get_residuals_transformed(self, fitted_model):
        """Test getting residuals in transformed scale."""
        residuals = fitted_model.get_residuals(in_original_scale=False)
        
        assert isinstance(residuals, np.ndarray)
    
    def test_get_residuals_original(self, fitted_model):
        """Test getting residuals in original scale."""
        residuals = fitted_model.get_residuals(in_original_scale=True)
        
        assert isinstance(residuals, np.ndarray)
        assert len(residuals) == 200


class TestImbalancePriceARMAStationarity:
    """Tests for stationarity checking."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model for testing."""
        data = generate_sample_data()
        model = ImbalancePriceARMA(p=1, q=1)
        model.fit(data)
        return model
    
    def test_check_stationarity_structure(self, fitted_model):
        """Test that stationarity check returns proper structure."""
        results = fitted_model.check_stationarity()
        
        assert 'adf_statistic' in results
        assert 'adf_pvalue' in results
        assert 'adf_is_stationary' in results
        assert 'kpss_statistic' in results
        assert 'kpss_pvalue' in results
        assert 'kpss_is_stationary' in results
    
    def test_check_stationarity_without_fit(self):
        """Test stationarity check without fitting raises error."""
        model = ImbalancePriceARMA(p=1, q=1)
        
        with pytest.raises(ValueError, match="No data available"):
            model.check_stationarity()
    
    def test_check_stationarity_with_custom_data(self, fitted_model):
        """Test stationarity check with custom data."""
        custom_data = np.random.randn(100)
        results = fitted_model.check_stationarity(data=custom_data)
        
        assert 'adf_pvalue' in results


class TestImbalancePriceARMAModelMetrics:
    """Tests for model metrics and summary."""
    
    @pytest.fixture
    def fitted_model(self):
        """Create a fitted model for testing."""
        data = generate_sample_data()
        model = ImbalancePriceARMA(p=1, q=1)
        model.fit(data)
        return model
    
    def test_aic(self, fitted_model):
        """Test AIC retrieval."""
        aic = fitted_model.aic
        
        assert isinstance(aic, float)
    
    def test_bic(self, fitted_model):
        """Test BIC retrieval."""
        bic = fitted_model.bic
        
        assert isinstance(bic, float)
    
    def test_summary(self, fitted_model):
        """Test summary retrieval."""
        summary = fitted_model.summary()
        
        assert isinstance(summary, str)
        assert len(summary) > 0
    
    def test_summary_without_fit(self):
        """Test summary without fitting."""
        model = ImbalancePriceARMA(p=1, q=1)
        summary = model.summary()
        
        assert summary == "Model not fitted yet"
    
    def test_aic_without_fit_raises_error(self):
        """Test AIC access without fitting raises error."""
        model = ImbalancePriceARMA(p=1, q=1)
        
        with pytest.raises(ValueError, match="Model must be fitted"):
            _ = model.aic
