import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch
from pydantic import ValidationError

from src.validation import (
    CreditRequest, validate_credit_request,
    validate_batch_data, detect_data_drift
)


class TestCreditRequestValidation:
    """Test suite for credit request validation."""

    def test_valid_credit_request(self):
        """Test creation of valid credit request."""
        request = CreditRequest(
            Age=35,
            CreditAmount=5000,
            Duration=24
        )
        
        assert request.Age == 35
        assert request.CreditAmount == 5000
        assert request.Duration == 24

    def test_age_validation_bounds(self):
        """Test age validation boundaries."""
        # Valid ages
        CreditRequest(Age=18, CreditAmount=1000, Duration=12)  # Minimum
        CreditRequest(Age=100, CreditAmount=1000, Duration=12)  # Maximum
        
        # Invalid ages
        with pytest.raises(ValidationError):
            CreditRequest(Age=17, CreditAmount=1000, Duration=12)  # Too young
        
        with pytest.raises(ValidationError):
            CreditRequest(Age=101, CreditAmount=1000, Duration=12)  # Too old

    def test_credit_amount_validation(self):
        """Test credit amount validation."""
        # Valid amounts
        CreditRequest(Age=30, CreditAmount=100, Duration=12)  # Minimum
        CreditRequest(Age=30, CreditAmount=100000, Duration=12)  # Maximum
        
        # Invalid amounts
        with pytest.raises(ValidationError):
            CreditRequest(Age=30, CreditAmount=99, Duration=12)  # Too low
        
        with pytest.raises(ValidationError):
            CreditRequest(Age=30, CreditAmount=100001, Duration=12)  # Too high

    def test_duration_validation(self):
        """Test duration validation."""
        # Valid durations
        CreditRequest(Age=30, CreditAmount=1000, Duration=1)  # Minimum
        CreditRequest(Age=30, CreditAmount=1000, Duration=72)  # Maximum
        
        # Invalid durations
        with pytest.raises(ValidationError):
            CreditRequest(Age=30, CreditAmount=1000, Duration=0)  # Too short
        
        with pytest.raises(ValidationError):
            CreditRequest(Age=30, CreditAmount=1000, Duration=73)  # Too long

    def test_business_rules_validation(self):
        """Test business rules validation."""
        # Young applicant warning
        request = CreditRequest(Age=20, CreditAmount=1000, Duration=12)
        result = request.validate_business_rules()
        
        assert not result["is_valid"]
        assert "Young applicant" in str(result["warnings"])
        
        # High credit amount relative to age
        request = CreditRequest(Age=25, CreditAmount=30000, Duration=12)
        result = request.validate_business_rules()
        
        assert not result["is_valid"]
        assert "Credit amount very high" in str(result["warnings"])
        
        # Long duration warning
        request = CreditRequest(Age=35, CreditAmount=5000, Duration=65)
        result = request.validate_business_rules()
        
        assert not result["is_valid"]
        assert "Very long credit duration" in str(result["warnings"])

    def test_optional_fields(self):
        """Test optional field handling."""
        request = CreditRequest(
            Age=35,
            CreditAmount=5000,
            Duration=24,
            CheckingStatus="A11",
            InstallmentCommitment=25.0
        )
        
        assert request.CheckingStatus == "A11"
        assert request.InstallmentCommitment == 25.0


class TestBatchValidation:
    """Test suite for batch data validation."""

    def test_valid_batch_data(self):
        """Test validation of clean batch data."""
        df = pd.DataFrame({
            'Age': [25, 30, 35],
            'CreditAmount': [1000, 2000, 3000],
            'Duration': [12, 24, 36]
        })
        
        report = validate_batch_data(df)
        
        assert report.total_records == 3
        assert report.quality_score > 90  # Should be high quality
        assert report.invalid_records == 0

    def test_batch_with_missing_values(self):
        """Test validation with missing values."""
        df = pd.DataFrame({
            'Age': [25, None, 35],
            'CreditAmount': [1000, 2000, None],
            'Duration': [12, 24, 36]
        })
        
        report = validate_batch_data(df)
        
        assert report.total_records == 3
        assert report.missing_values['Age'] == 1
        assert report.missing_values['CreditAmount'] == 1
        assert len(report.recommendations) > 0

    def test_batch_with_outliers(self):
        """Test outlier detection."""
        # Create data with obvious outliers
        normal_ages = [25, 30, 35, 40, 45]
        outlier_ages = [150, 200]  # Impossible ages
        
        df = pd.DataFrame({
            'Age': normal_ages + outlier_ages,
            'CreditAmount': [1000] * 7,
            'Duration': [12] * 7
        })
        
        report = validate_batch_data(df)
        
        assert 'Age' in report.outliers
        assert len(report.outliers['Age']) > 0


class TestDriftDetection:
    """Test suite for data drift detection."""

    def test_no_drift_detection(self):
        """Test when no drift is present."""
        # Create similar distributions
        np.random.seed(42)
        reference_df = pd.DataFrame({
            'Age': np.random.normal(35, 10, 1000),
            'CreditAmount': np.random.normal(5000, 1000, 1000)
        })
        
        current_df = pd.DataFrame({
            'Age': np.random.normal(35, 10, 500),
            'CreditAmount': np.random.normal(5000, 1000, 500)
        })
        
        result = detect_data_drift(current_df, reference_df)
        
        assert result['alert_level'] == 'LOW'
        assert len(result['drifted_features']) == 0

    def test_drift_detection(self):
        """Test when drift is present."""
        np.random.seed(42)
        reference_df = pd.DataFrame({
            'Age': np.random.normal(35, 10, 1000),
            'CreditAmount': np.random.normal(5000, 1000, 1000)
        })
        
        # Shift the distribution significantly
        current_df = pd.DataFrame({
            'Age': np.random.normal(50, 10, 500),  # Mean shift
            'CreditAmount': np.random.normal(8000, 1000, 500)  # Mean shift
        })
        
        result = detect_data_drift(current_df, reference_df)
        
        assert result['alert_level'] in ['MEDIUM', 'HIGH']
        assert len(result['drifted_features']) > 0


class TestValidationIntegration:
    """Integration tests for validation components."""

    def test_validate_credit_request_integration(self):
        """Test the main validation function."""
        request = CreditRequest(Age=35, CreditAmount=5000, Duration=24)
        
        # Should not raise any exceptions
        validated = validate_credit_request(request)
        
        assert validated.Age == 35
        assert validated.CreditAmount == 5000
        assert validated.Duration == 24

    def test_validate_credit_request_with_warnings(self):
        """Test validation with business rule warnings."""
        request = CreditRequest(Age=20, CreditAmount=25000, Duration=65)
        
        # Should not raise exceptions but log warnings
        with patch('src.validation.logger') as mock_logger:
            validated = validate_credit_request(request)
            
            assert validated is not None
            mock_logger.warning.assert_called()  # Should log warnings