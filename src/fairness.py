import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class FairnessEvaluator:
    """Comprehensive fairness evaluation for credit scoring models."""
    
    def __init__(self, sensitive_features: List[str]):
        """
        Initialize fairness evaluator.
        
        Args:
            sensitive_features: List of sensitive/protected feature names
        """
        self.sensitive_features = sensitive_features
    
    def evaluate_group_fairness(self, 
                               y_true: np.ndarray, 
                               y_pred: np.ndarray, 
                               y_scores: np.ndarray,
                               df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate group fairness metrics across protected groups.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels  
            y_scores: Prediction scores/probabilities
            df: DataFrame with features including sensitive attributes
            
        Returns:
            Dictionary with fairness metrics
        """
        fairness_metrics = {}
        
        for feature in self.sensitive_features:
            if feature not in df.columns:
                continue
                
            feature_metrics = {}
            unique_values = df[feature].unique()
            
            # Calculate metrics for each group
            group_metrics = {}
            for value in unique_values:
                mask = df[feature] == value
                if mask.sum() == 0:
                    continue
                    
                group_y_true = y_true[mask]
                group_y_pred = y_pred[mask]
                group_y_scores = y_scores[mask]
                
                group_metrics[str(value)] = self._calculate_group_metrics(
                    group_y_true, group_y_pred, group_y_scores
                )
            
            # Calculate fairness metrics
            feature_metrics['group_metrics'] = group_metrics
            feature_metrics['fairness_metrics'] = self._calculate_fairness_metrics(group_metrics)
            
            fairness_metrics[feature] = feature_metrics
        
        return fairness_metrics
    
    def _calculate_group_metrics(self, 
                                y_true: np.ndarray, 
                                y_pred: np.ndarray, 
                                y_scores: np.ndarray) -> Dict[str, float]:
        """Calculate metrics for a single group."""
        if len(y_true) == 0:
            return {}
        
        # Basic metrics
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        
        # Rates
        positive_rate = np.mean(y_pred)  # Selection rate
        true_positive_rate = tp / (tp + fn) if (tp + fn) > 0 else 0
        false_positive_rate = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        return {
            'count': len(y_true),
            'positive_rate': positive_rate,
            'true_positive_rate': true_positive_rate,
            'false_positive_rate': false_positive_rate,
            'precision': precision,
            'base_rate': np.mean(y_true),  # Actual positive rate in group
            'mean_score': np.mean(y_scores)
        }
    
    def _calculate_fairness_metrics(self, group_metrics: Dict) -> Dict[str, float]:
        """Calculate fairness metrics from group metrics."""
        groups = list(group_metrics.keys())
        if len(groups) < 2:
            return {}
        
        # Extract metrics for comparison
        positive_rates = [group_metrics[g]['positive_rate'] for g in groups]
        tpr_rates = [group_metrics[g]['true_positive_rate'] for g in groups]
        fpr_rates = [group_metrics[g]['false_positive_rate'] for g in groups]
        precisions = [group_metrics[g]['precision'] for g in groups]
        
        # Demographic Parity (Statistical Parity)
        demographic_parity_diff = max(positive_rates) - min(positive_rates)
        demographic_parity_ratio = min(positive_rates) / max(positive_rates) if max(positive_rates) > 0 else 0
        
        # Equalized Odds (TPR and FPR should be similar across groups)
        tpr_diff = max(tpr_rates) - min(tpr_rates)
        fpr_diff = max(fpr_rates) - min(fpr_rates)
        equalized_odds_diff = max(tpr_diff, fpr_diff)
        
        # Equal Opportunity (TPR should be similar across groups)
        equal_opportunity_diff = tpr_diff
        
        # Precision Parity
        precision_diff = max(precisions) - min(precisions)
        
        return {
            'demographic_parity_difference': demographic_parity_diff,
            'demographic_parity_ratio': demographic_parity_ratio,
            'equalized_odds_difference': equalized_odds_diff,
            'equal_opportunity_difference': equal_opportunity_diff,
            'precision_parity_difference': precision_diff
        }
    
    def evaluate_individual_fairness(self, 
                                   df: pd.DataFrame, 
                                   y_scores: np.ndarray,
                                   distance_threshold: float = 0.1) -> Dict[str, float]:
        """
        Evaluate individual fairness using similarity-based approach.
        
        Args:
            df: DataFrame with features
            y_scores: Prediction scores
            distance_threshold: Threshold for considering individuals as similar
            
        Returns:
            Individual fairness metrics
        """
        from sklearn.metrics.pairwise import euclidean_distances
        from sklearn.preprocessing import StandardScaler
        
        # Use only non-sensitive features for similarity
        feature_cols = [col for col in df.columns if col not in self.sensitive_features]
        X = df[feature_cols].select_dtypes(include=[np.number])
        
        if X.empty:
            return {'individual_fairness_violation_rate': 0.0}
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X.fillna(X.mean()))
        
        # Calculate pairwise distances
        distances = euclidean_distances(X_scaled)
        
        violations = 0
        total_pairs = 0
        
        n_samples = min(1000, len(X_scaled))  # Limit for computational efficiency
        indices = np.random.choice(len(X_scaled), n_samples, replace=False)
        
        for i, idx_i in enumerate(indices):
            for j, idx_j in enumerate(indices[i+1:], i+1):
                if distances[idx_i, idx_j] <= distance_threshold:
                    # Similar individuals should have similar predictions
                    score_diff = abs(y_scores[idx_i] - y_scores[idx_j])
                    if score_diff > distance_threshold:
                        violations += 1
                    total_pairs += 1
        
        violation_rate = violations / total_pairs if total_pairs > 0 else 0
        
        return {
            'individual_fairness_violation_rate': violation_rate,
            'total_similar_pairs_evaluated': total_pairs
        }
    
    def generate_fairness_report(self, 
                                y_true: np.ndarray,
                                y_pred: np.ndarray, 
                                y_scores: np.ndarray,
                                df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive fairness report."""
        
        # Group fairness evaluation
        group_fairness = self.evaluate_group_fairness(y_true, y_pred, y_scores, df)
        
        # Individual fairness evaluation
        individual_fairness = self.evaluate_individual_fairness(df, y_scores)
        
        # Overall fairness assessment
        fairness_violations = []
        fairness_score = 100.0  # Start with perfect score
        
        for feature, metrics in group_fairness.items():
            if 'fairness_metrics' not in metrics:
                continue
                
            fm = metrics['fairness_metrics']
            
            # Check thresholds (commonly used in literature)
            if fm.get('demographic_parity_difference', 0) > 0.1:
                fairness_violations.append(f"Demographic parity violation in {feature}")
                fairness_score -= 15
            
            if fm.get('equalized_odds_difference', 0) > 0.1:
                fairness_violations.append(f"Equalized odds violation in {feature}")
                fairness_score -= 15
            
            if fm.get('equal_opportunity_difference', 0) > 0.1:
                fairness_violations.append(f"Equal opportunity violation in {feature}")
                fairness_score -= 10
        
        # Individual fairness penalty
        if individual_fairness.get('individual_fairness_violation_rate', 0) > 0.2:
            fairness_violations.append("High individual fairness violation rate")
            fairness_score -= 20
        
        fairness_score = max(0, fairness_score)  # Don't go below 0
        
        return {
            'group_fairness': group_fairness,
            'individual_fairness': individual_fairness,
            'fairness_violations': fairness_violations,
            'overall_fairness_score': fairness_score,
            'is_fair': len(fairness_violations) == 0,
            'recommendations': self._generate_recommendations(fairness_violations, group_fairness)
        }
    
    def _generate_recommendations(self, 
                                violations: List[str], 
                                group_fairness: Dict) -> List[str]:
        """Generate actionable recommendations based on fairness violations."""
        recommendations = []
        
        if not violations:
            recommendations.append("Model appears to be fair across evaluated metrics.")
            return recommendations
        
        if any("Demographic parity" in v for v in violations):
            recommendations.append(
                "Consider post-processing techniques to equalize selection rates across groups."
            )
        
        if any("Equalized odds" in v for v in violations):
            recommendations.append(
                "Consider adversarial debiasing or constraint-based optimization during training."
            )
        
        if any("individual fairness" in v.lower() for v in violations):
            recommendations.append(
                "Review feature engineering to ensure similar individuals are treated similarly."
            )
        
        # Feature-specific recommendations
        for feature, metrics in group_fairness.items():
            if 'group_metrics' in metrics:
                group_sizes = [m['count'] for m in metrics['group_metrics'].values()]
                if min(group_sizes) / max(group_sizes) < 0.1:  # Highly imbalanced groups
                    recommendations.append(
                        f"Consider stratified sampling for {feature} to balance group representation."
                    )
        
        recommendations.append(
            "Regularly monitor fairness metrics in production to detect drift in model fairness."
        )
        
        return recommendations

def evaluate_model_fairness(test_df: pd.DataFrame, 
                          predictions: np.ndarray,
                          scores: np.ndarray,
                          sensitive_features: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Convenience function to evaluate model fairness.
    
    Args:
        test_df: Test dataframe with features and true labels
        predictions: Binary predictions
        scores: Prediction scores/probabilities  
        sensitive_features: List of sensitive feature names
        
    Returns:
        Fairness evaluation results
    """
    if sensitive_features is None:
        # Common sensitive features in credit scoring
        sensitive_features = ['PersonalStatus', 'Age', 'ForeignWorker']
        # Filter to only those present in the dataset
        sensitive_features = [f for f in sensitive_features if f in test_df.columns]
    
    if not sensitive_features:
        return {
            'warning': 'No sensitive features found for fairness evaluation',
            'fairness_score': 100.0,
            'is_fair': True
        }
    
    evaluator = FairnessEvaluator(sensitive_features)
    
    y_true = test_df['y'].values if 'y' in test_df.columns else np.array([])
    
    if len(y_true) == 0:
        return {
            'warning': 'No true labels found for fairness evaluation', 
            'fairness_score': 100.0,
            'is_fair': True
        }
    
    return evaluator.generate_fairness_report(
        y_true=y_true,
        y_pred=predictions,
        y_scores=scores, 
        df=test_df
    )