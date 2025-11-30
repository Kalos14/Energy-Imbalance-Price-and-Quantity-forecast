"""
Tests for the transforms module.
"""

import numpy as np
import pytest

from imbalance_price_model.transforms import arcsinh_transform, inverse_arcsinh_transform


class TestArcsinhTransform:
    """Tests for arcsinh_transform function."""
    
    def test_basic_transformation(self):
        """Test that basic transformation works."""
        data = np.array([100.0, -50.0, 200.0, -100.0, 150.0])
        transformed, std = arcsinh_transform(data)
        
        assert transformed.shape == data.shape
        assert isinstance(std, float)
        assert std > 0
    
    def test_transformation_formula(self):
        """Test that the transformation follows arcsinh(x/std)/std formula."""
        data = np.array([10.0, 20.0, 30.0, -10.0, -20.0])
        transformed, std = arcsinh_transform(data)
        
        # Manually compute expected transformation
        expected = np.arcsinh(data / std) / std
        
        np.testing.assert_array_almost_equal(transformed, expected)
    
    def test_with_custom_std(self):
        """Test transformation with custom std value."""
        data = np.array([100.0, 200.0, 300.0])
        custom_std = 50.0
        
        transformed, returned_std = arcsinh_transform(data, std=custom_std)
        
        assert returned_std == custom_std
        expected = np.arcsinh(data / custom_std) / custom_std
        np.testing.assert_array_almost_equal(transformed, expected)
    
    def test_handles_negative_values(self):
        """Test that negative values are handled correctly."""
        data = np.array([-100.0, -50.0, 0.0, 50.0, 100.0])
        transformed, std = arcsinh_transform(data)
        
        # arcsinh is an odd function, so negative inputs give negative outputs
        # After scaling, the sign should be preserved
        assert transformed[0] < 0  # -100 should give negative
        assert transformed[4] > 0  # 100 should give positive
        assert np.isclose(transformed[2], 0, atol=1e-10)  # 0 should give ~0
    
    def test_handles_zero_std_data(self):
        """Test handling of constant data (zero std)."""
        data = np.array([5.0, 5.0, 5.0, 5.0])
        transformed, std = arcsinh_transform(data)
        
        # Should use std=1.0 to avoid division by zero
        assert std == 1.0
    
    def test_preserves_array_type(self):
        """Test that output is numpy array with float64 dtype."""
        data = [1, 2, 3, 4, 5]  # List input
        transformed, std = arcsinh_transform(data)
        
        assert isinstance(transformed, np.ndarray)
        assert transformed.dtype == np.float64


class TestInverseArcsinhTransform:
    """Tests for inverse_arcsinh_transform function."""
    
    def test_inverse_recovers_original(self):
        """Test that inverse transformation recovers original data."""
        original = np.array([100.0, -50.0, 200.0, -100.0, 150.0])
        transformed, std = arcsinh_transform(original)
        recovered = inverse_arcsinh_transform(transformed, std)
        
        np.testing.assert_array_almost_equal(recovered, original)
    
    def test_inverse_formula(self):
        """Test that inverse follows std * sinh(std * x) formula."""
        transformed = np.array([0.1, -0.05, 0.15])
        std = 100.0
        
        recovered = inverse_arcsinh_transform(transformed, std)
        expected = std * np.sinh(std * transformed)
        
        np.testing.assert_array_almost_equal(recovered, expected)
    
    def test_roundtrip_various_scales(self):
        """Test roundtrip transformation for various data scales."""
        # Small values
        small = np.array([0.01, 0.02, -0.01, 0.005])
        transformed_small, std_small = arcsinh_transform(small)
        recovered_small = inverse_arcsinh_transform(transformed_small, std_small)
        np.testing.assert_array_almost_equal(recovered_small, small)
        
        # Large values
        large = np.array([1000.0, -2000.0, 5000.0, -3000.0])
        transformed_large, std_large = arcsinh_transform(large)
        recovered_large = inverse_arcsinh_transform(transformed_large, std_large)
        np.testing.assert_array_almost_equal(recovered_large, large)
    
    def test_preserves_array_type(self):
        """Test that output is numpy array with float64 dtype."""
        transformed = [0.1, 0.2, 0.3]
        recovered = inverse_arcsinh_transform(transformed, std=10.0)
        
        assert isinstance(recovered, np.ndarray)
        assert recovered.dtype == np.float64


class TestTransformProperties:
    """Tests for mathematical properties of the transform."""
    
    def test_monotonicity(self):
        """Test that transform preserves order (monotonic)."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        transformed, _ = arcsinh_transform(data)
        
        # Should be strictly increasing
        assert np.all(np.diff(transformed) > 0)
    
    def test_odd_function_property(self):
        """Test that transform is an odd function (f(-x) = -f(x))."""
        data = np.array([1.0, 2.0, 3.0])
        neg_data = -data
        
        transformed_pos, std = arcsinh_transform(data)
        transformed_neg, _ = arcsinh_transform(neg_data, std=std)
        
        np.testing.assert_array_almost_equal(transformed_neg, -transformed_pos)
    
    def test_reduces_heavy_tails(self):
        """Test that transformation reduces the impact of extreme values."""
        # Data with heavy tails
        data = np.array([1.0, 2.0, 100.0, 3.0, 4.0])  # 100 is an outlier
        transformed, std = arcsinh_transform(data)
        
        # The ratio of max to median should be smaller after transformation
        original_ratio = np.max(np.abs(data)) / np.median(np.abs(data))
        transformed_ratio = np.max(np.abs(transformed)) / np.median(np.abs(transformed))
        
        assert transformed_ratio < original_ratio
