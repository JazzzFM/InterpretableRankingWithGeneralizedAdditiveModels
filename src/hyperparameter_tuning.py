import optuna
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score
import mlflow
import logging
from pygam import LogisticGAM, s, f

logger = logging.getLogger(__name__)

class GAMTuner:
    """Advanced hyperparameter tuning for GAM models using Optuna."""
    
    def __init__(self, 
                 n_trials: int = 100,
                 cv_folds: int = 5,
                 random_state: int = 42,
                 study_name: Optional[str] = None):
        """
        Initialize GAM tuner.
        
        Args:
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
            study_name: Name of the Optuna study
        """
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.study_name = study_name or f"gam_tuning_{random_state}"
        
        # Initialize Optuna study
        self.study = optuna.create_study(
            direction="maximize",
            study_name=self.study_name,
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        
        # Store data for optimization
        self.X = None
        self.y = None
        self.numeric_features = None
        self.categorical_features = None
        
    def objective(self, trial: optuna.Trial) -> float:
        """
        Objective function for Optuna optimization.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Cross-validation AUC score
        """
        try:
            # Build GAM terms with hyperparameters
            terms = []
            
            # Optimize numeric feature splines
            for i, feature in enumerate(self.numeric_features):
                # Suggest regularization parameter (lambda)
                lam = trial.suggest_float(f"lam_numeric_{i}", 1e-3, 1e2, log=True)
                
                # Suggest number of splines (optional advanced parameter)
                n_splines = trial.suggest_int(f"n_splines_{i}", 5, 30)
                
                terms.append(s(i, lam=lam, n_splines=n_splines))
            
            # Optimize categorical feature regularization
            for i, feature in enumerate(self.categorical_features):
                feature_idx = len(self.numeric_features) + i
                lam = trial.suggest_float(f"lam_categorical_{i}", 1e-3, 1e2, log=True)
                
                terms.append(f(feature_idx, lam=lam))
            
            # Create GAM model
            gam = LogisticGAM(terms)
            
            # Perform cross-validation
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            
            cv_scores = []
            for train_idx, val_idx in cv.split(self.X, self.y):
                X_train, X_val = self.X[train_idx], self.X[val_idx]
                y_train, y_val = self.y[train_idx], self.y[val_idx]
                
                # Fit model
                gam.fit(X_train, y_train)
                
                # Predict probabilities
                y_pred_proba = gam.predict_proba(X_val)
                
                # Calculate AUC
                auc = roc_auc_score(y_val, y_pred_proba)
                cv_scores.append(auc)
            
            mean_auc = np.mean(cv_scores)
            
            # Log trial results with MLflow (if available)
            try:
                with mlflow.start_run(nested=True):
                    mlflow.log_params(trial.params)
                    mlflow.log_metric("cv_auc", mean_auc)
            except Exception:
                pass  # MLflow might not be available
            
            return mean_auc
            
        except Exception as e:
            logger.warning(f"Trial failed: {e}")
            return 0.0  # Return poor score for failed trials
    
    def optimize(self, 
                 X: np.ndarray, 
                 y: np.ndarray,
                 numeric_features: list,
                 categorical_features: list) -> Dict[str, Any]:
        """
        Optimize GAM hyperparameters.
        
        Args:
            X: Feature matrix
            y: Target vector
            numeric_features: List of numeric feature names
            categorical_features: List of categorical feature names
            
        Returns:
            Dictionary with optimization results
        """
        logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        
        # Store data for objective function
        self.X = X
        self.y = y
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        
        # Add pruning callback for early stopping of unpromising trials
        pruner_callback = optuna.integration.MLflowCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            metric_name="cv_auc"
        ) if mlflow.active_run() else None
        
        # Optimize
        self.study.optimize(
            self.objective, 
            n_trials=self.n_trials,
            callbacks=[pruner_callback] if pruner_callback else None
        )
        
        # Get best parameters
        best_params = self.study.best_params
        best_value = self.study.best_value
        
        logger.info(f"Optimization completed. Best AUC: {best_value:.4f}")
        
        # Build optimized terms
        optimized_terms = self._build_optimized_terms(best_params)
        
        return {
            'best_params': best_params,
            'best_value': best_value,
            'optimized_terms': optimized_terms,
            'study': self.study,
            'optimization_history': self._get_optimization_history()
        }
    
    def _build_optimized_terms(self, best_params: Dict[str, Any]) -> list:
        """Build GAM terms with optimized parameters."""
        terms = []
        
        # Numeric features
        for i, feature in enumerate(self.numeric_features):
            lam = best_params[f"lam_numeric_{i}"]
            n_splines = best_params.get(f"n_splines_{i}", 20)
            terms.append(s(i, lam=lam, n_splines=n_splines))
        
        # Categorical features
        for i, feature in enumerate(self.categorical_features):
            feature_idx = len(self.numeric_features) + i
            lam = best_params[f"lam_categorical_{i}"]
            terms.append(f(feature_idx, lam=lam))
        
        return terms
    
    def _get_optimization_history(self) -> Dict[str, Any]:
        """Get optimization history for analysis."""
        trials = self.study.trials
        
        history = {
            'trial_numbers': [t.number for t in trials],
            'values': [t.value if t.value else 0.0 for t in trials],
            'states': [t.state.name for t in trials],
            'durations': [t.duration.total_seconds() if t.duration else 0 for t in trials]
        }
        
        return history
    
    def plot_optimization_history(self, save_path: Optional[str] = None) -> None:
        """Plot optimization history."""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            
            # Optimization history
            optuna.visualization.matplotlib.plot_optimization_history(self.study, ax=axes[0, 0])
            axes[0, 0].set_title("Optimization History")
            
            # Parameter importances
            try:
                optuna.visualization.matplotlib.plot_param_importances(self.study, ax=axes[0, 1])
                axes[0, 1].set_title("Parameter Importances")
            except Exception:
                axes[0, 1].text(0.5, 0.5, "Parameter importance\nnot available", ha='center', va='center')
            
            # Parallel coordinate plot
            try:
                optuna.visualization.matplotlib.plot_parallel_coordinate(self.study, ax=axes[1, 0])
                axes[1, 0].set_title("Parallel Coordinate")
            except Exception:
                axes[1, 0].text(0.5, 0.5, "Parallel coordinate\nnot available", ha='center', va='center')
            
            # Slice plot for top parameter
            try:
                importances = optuna.importance.get_param_importances(self.study)
                top_param = max(importances.keys(), key=lambda k: importances[k])
                optuna.visualization.matplotlib.plot_slice(self.study, [top_param], ax=axes[1, 1])
                axes[1, 1].set_title(f"Slice Plot: {top_param}")
            except Exception:
                axes[1, 1].text(0.5, 0.5, "Slice plot\nnot available", ha='center', va='center')
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logger.info(f"Optimization plots saved to {save_path}")
            
            plt.show()
            
        except ImportError:
            logger.warning("Matplotlib not available for plotting")

def optimize_gam_hyperparameters(X: np.ndarray,
                                y: np.ndarray,
                                numeric_features: list,
                                categorical_features: list,
                                n_trials: int = 100,
                                cv_folds: int = 5,
                                random_state: int = 42) -> Dict[str, Any]:
    """
    Convenience function for GAM hyperparameter optimization.
    
    Args:
        X: Feature matrix
        y: Target vector
        numeric_features: List of numeric feature names
        categorical_features: List of categorical feature names
        n_trials: Number of optimization trials
        cv_folds: Number of CV folds
        random_state: Random state
        
    Returns:
        Optimization results
    """
    tuner = GAMTuner(
        n_trials=n_trials,
        cv_folds=cv_folds,
        random_state=random_state
    )
    
    return tuner.optimize(X, y, numeric_features, categorical_features)

# Example usage and integration with existing model training
class OptimizedGAMTrainer:
    """Enhanced GAM trainer with hyperparameter optimization."""
    
    def __init__(self, optimize: bool = True, n_trials: int = 50):
        """
        Initialize optimized GAM trainer.
        
        Args:
            optimize: Whether to perform hyperparameter optimization
            n_trials: Number of optimization trials
        """
        self.optimize = optimize
        self.n_trials = n_trials
        
    def train(self, 
              X: np.ndarray, 
              y: np.ndarray,
              numeric_features: list,
              categorical_features: list) -> Tuple[LogisticGAM, Dict[str, Any]]:
        """
        Train GAM with optional hyperparameter optimization.
        
        Returns:
            Trained GAM model and optimization results
        """
        if self.optimize:
            logger.info("Training GAM with hyperparameter optimization")
            
            optimization_results = optimize_gam_hyperparameters(
                X=X,
                y=y,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
                n_trials=self.n_trials
            )
            
            # Train final model with optimized parameters
            optimized_terms = optimization_results['optimized_terms']
            gam = LogisticGAM(optimized_terms)
            gam.fit(X, y)
            
            return gam, optimization_results
            
        else:
            logger.info("Training GAM with default parameters")
            
            # Build default terms
            terms = []
            for i in range(len(numeric_features)):
                terms.append(s(i))
            for i in range(len(categorical_features)):
                terms.append(f(len(numeric_features) + i))
            
            gam = LogisticGAM(terms)
            gam.fit(X, y)
            
            return gam, {'optimization_performed': False}