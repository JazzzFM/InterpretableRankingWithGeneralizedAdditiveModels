from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class CreditRequest(BaseModel):
    """Enhanced credit request with comprehensive validation."""
    
    Age: float = Field(..., ge=18, le=100, description="Age of applicant")
    CreditAmount: float = Field(..., ge=100, le=100000, description="Credit amount requested") 
    Duration: float = Field(..., ge=1, le=72, description="Duration in months")
    
    # Additional fields that might be in the German Credit dataset
    CheckingStatus: Optional[str] = Field(None, description="Checking account status")
    CreditHistory: Optional[str] = Field(None, description="Credit history")
    Purpose: Optional[str] = Field(None, description="Purpose of credit")
    SavingsStatus: Optional[str] = Field(None, description="Savings account status")
    Employment: Optional[str] = Field(None, description="Employment status")
    InstallmentCommitment: Optional[float] = Field(None, ge=0, le=100, description="Installment rate in percentage")
    PersonalStatus: Optional[str] = Field(None, description="Personal status and sex")
    OtherParties: Optional[str] = Field(None, description="Other parties")
    Residence: Optional[float] = Field(None, ge=0, le=10, description="Residence since")
    Property: Optional[str] = Field(None, description="Property")
    OtherPaymentPlans: Optional[str] = Field(None, description="Other payment plans")
    Housing: Optional[str] = Field(None, description="Housing")
    ExistingCredits: Optional[float] = Field(None, ge=1, le=10, description="Number of existing credits")
    Job: Optional[str] = Field(None, description="Job")
    Liable: Optional[float] = Field(None, ge=1, le=5, description="Number of liable people")
    Telephone: Optional[str] = Field(None, description="Telephone")
    ForeignWorker: Optional[str] = Field(None, description="Foreign worker")

    @validator('Age')
    def validate_age(cls, v):
        """Validate age is reasonable."""
        if not 18 <= v <= 100:
            raise ValueError('Age must be between 18 and 100')
        return v

    @validator('CreditAmount')
    def validate_credit_amount(cls, v):
        """Validate credit amount is reasonable."""
        if not 100 <= v <= 100000:
            raise ValueError('Credit amount must be between 100 and 100,000')
        return v

    @validator('Duration')
    def validate_duration(cls, v):
        """Validate duration is reasonable."""
        if not 1 <= v <= 72:
            raise ValueError('Duration must be between 1 and 72 months')
        return v

    @validator('InstallmentCommitment')
    def validate_installment_commitment(cls, v):
        """Validate installment commitment percentage."""
        if v is not None and not 0 <= v <= 100:
            raise ValueError('Installment commitment must be between 0 and 100%')
        return v

    def validate_business_rules(self) -> Dict[str, Any]:
        """Apply business rule validations."""
        warnings = []
        
        # High risk indicators
        if self.Age < 21:
            warnings.append("Young applicant - higher risk")
        
        if self.CreditAmount > self.Age * 1000:
            warnings.append("Credit amount very high relative to age")
            
        if self.Duration > 60:
            warnings.append("Very long credit duration")
            
        # Debt-to-income approximation (if installment commitment available)
        if self.InstallmentCommitment and self.InstallmentCommitment > 40:
            warnings.append("High installment commitment percentage")
            
        return {
            "is_valid": len(warnings) == 0,
            "warnings": warnings,
            "risk_score": min(len(warnings), 5)  # Simple risk score 0-5
        }

class DataQualityReport(BaseModel):
    """Data quality assessment report."""
    
    total_records: int
    missing_values: Dict[str, int]
    outliers: Dict[str, List[float]]
    invalid_records: int
    quality_score: float
    recommendations: List[str]
    timestamp: datetime = Field(default_factory=datetime.now)

def validate_credit_request(request: CreditRequest) -> CreditRequest:
    """Validate a single credit request with enhanced checks."""
    
    try:
        # Business rule validation
        validation_result = request.validate_business_rules()
        
        # Log validation results
        if not validation_result["is_valid"]:
            logger.warning(f"Business rule warnings: {validation_result['warnings']}")
        
        # Could add additional validations here
        # - Check against blacklists
        # - Verify data consistency
        # - Apply statistical outlier detection
        
        return request
        
    except Exception as e:
        logger.error(f"Validation error: {e}")
        raise ValueError(f"Invalid request: {e}")

def validate_batch_data(df: pd.DataFrame) -> DataQualityReport:
    """Validate a batch of data and return quality report."""
    
    total_records = len(df)
    missing_values = df.isnull().sum().to_dict()
    invalid_records = 0
    recommendations = []
    
    # Check for outliers using IQR method
    outliers = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outlier_values = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].tolist()
        if outlier_values:
            outliers[col] = outlier_values[:10]  # Limit to first 10 outliers
    
    # Count invalid records
    try:
        for _, row in df.iterrows():
            try:
                CreditRequest(**row.to_dict())
            except Exception:
                invalid_records += 1
    except Exception as e:
        logger.error(f"Error validating batch data: {e}")
    
    # Generate recommendations
    missing_pct = sum(missing_values.values()) / (len(df.columns) * total_records) * 100
    
    if missing_pct > 5:
        recommendations.append(f"High missing data rate: {missing_pct:.1f}%")
    
    if len(outliers) > 0:
        recommendations.append(f"Outliers detected in {len(outliers)} columns")
        
    if invalid_records > 0:
        recommendations.append(f"{invalid_records} records failed validation")
    
    # Calculate quality score (0-100)
    quality_score = max(0, 100 - missing_pct - (invalid_records / total_records * 100) - len(outliers) * 2)
    
    return DataQualityReport(
        total_records=total_records,
        missing_values=missing_values,
        outliers=outliers,
        invalid_records=invalid_records,
        quality_score=quality_score,
        recommendations=recommendations
    )

def detect_data_drift(current_df: pd.DataFrame, reference_df: pd.DataFrame) -> Dict[str, Any]:
    """Detect data drift between current and reference datasets."""
    
    drift_results = {}
    
    # Compare numeric features using statistical tests
    from scipy import stats
    
    numeric_cols = current_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col in reference_df.columns:
            # Kolmogorov-Smirnov test for distribution comparison
            ks_stat, p_value = stats.ks_2samp(current_df[col].dropna(), reference_df[col].dropna())
            
            drift_results[col] = {
                'ks_statistic': ks_stat,
                'p_value': p_value,
                'drift_detected': p_value < 0.05,  # 5% significance level
                'current_mean': current_df[col].mean(),
                'reference_mean': reference_df[col].mean(),
                'mean_shift': abs(current_df[col].mean() - reference_df[col].mean())
            }
    
    # Summary
    drifted_features = [col for col, result in drift_results.items() if result['drift_detected']]
    
    return {
        'drift_results': drift_results,
        'drifted_features': drifted_features,
        'drift_score': len(drifted_features) / len(drift_results) if drift_results else 0,
        'alert_level': 'HIGH' if len(drifted_features) > len(drift_results) * 0.3 else 'MEDIUM' if drifted_features else 'LOW'
    }