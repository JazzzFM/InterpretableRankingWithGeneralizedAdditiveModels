import time
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from collections import defaultdict, deque
import threading
from dataclasses import dataclass, asdict
import os

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class PredictionMetric:
    """Stores metrics for a single prediction request."""
    timestamp: datetime
    user: str
    model_version: str
    prediction_time: float
    input_features: Dict[str, Any]
    prediction: float
    decision: str
    request_id: str

@dataclass
class SystemMetric:
    """Stores system performance metrics."""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    active_requests: int
    total_requests: int
    error_rate: float

class MetricsCollector:
    """Thread-safe metrics collector with in-memory storage."""
    
    def __init__(self, max_predictions: int = 10000):
        self._lock = threading.Lock()
        self.predictions: deque = deque(maxlen=max_predictions)
        self.system_metrics: deque = deque(maxlen=1000)
        self.error_counts = defaultdict(int)
        self.request_counts = defaultdict(int)
        self.response_times = deque(maxlen=1000)
        
    def log_prediction(self, metric: PredictionMetric):
        """Log a prediction metric."""
        with self._lock:
            self.predictions.append(metric)
            self.request_counts[metric.user] += 1
            self.response_times.append(metric.prediction_time)
            
            # Structured logging
            logger.info("Prediction made", extra={
                "user": metric.user,
                "prediction": metric.prediction,
                "decision": metric.decision,
                "prediction_time": metric.prediction_time,
                "request_id": metric.request_id
            })
    
    def log_error(self, error_type: str, user: str, details: str):
        """Log an error occurrence."""
        with self._lock:
            self.error_counts[error_type] += 1
            
            logger.error("API error", extra={
                "error_type": error_type,
                "user": user,
                "details": details,
                "timestamp": datetime.now().isoformat()
            })
    
    def log_system_metric(self, metric: SystemMetric):
        """Log system performance metrics."""
        with self._lock:
            self.system_metrics.append(metric)
            
    def get_metrics_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Get metrics summary for the specified time window."""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
            
            # Filter recent predictions
            recent_predictions = [
                p for p in self.predictions 
                if p.timestamp > cutoff_time
            ]
            
            # Calculate summary statistics
            total_predictions = len(recent_predictions)
            avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
            
            # Decision distribution
            decisions = [p.decision for p in recent_predictions]
            decision_counts = {
                "approve": decisions.count("approve"),
                "review": decisions.count("review")
            }
            
            # Prediction distribution
            predictions = [p.prediction for p in recent_predictions]
            avg_prediction = sum(predictions) / len(predictions) if predictions else 0
            
            # User activity
            user_activity = defaultdict(int)
            for p in recent_predictions:
                user_activity[p.user] += 1
                
            # Error summary
            total_errors = sum(self.error_counts.values())
            error_rate = (total_errors / (total_predictions + total_errors)) * 100 if (total_predictions + total_errors) > 0 else 0
            
            return {
                "time_window_minutes": window_minutes,
                "timestamp": datetime.now().isoformat(),
                "predictions": {
                    "total": total_predictions,
                    "decisions": decision_counts,
                    "avg_probability": round(avg_prediction, 3),
                    "avg_response_time_ms": round(avg_response_time * 1000, 2)
                },
                "users": {
                    "active_users": len(user_activity),
                    "user_activity": dict(user_activity)
                },
                "errors": {
                    "total": total_errors,
                    "rate_percent": round(error_rate, 2),
                    "by_type": dict(self.error_counts)
                },
                "system": {
                    "uptime_hours": self._get_uptime_hours(),
                    "total_requests": sum(self.request_counts.values())
                }
            }
    
    def get_prediction_distribution(self, bins: int = 10) -> Dict[str, List[float]]:
        """Get prediction probability distribution."""
        with self._lock:
            predictions = [p.prediction for p in self.predictions]
            
            if not predictions:
                return {"bins": [], "counts": []}
            
            # Create histogram
            import numpy as np
            counts, bin_edges = np.histogram(predictions, bins=bins, range=(0, 1))
            
            return {
                "bins": bin_edges.tolist(),
                "counts": counts.tolist(),
                "total_samples": len(predictions)
            }
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get model performance metrics."""
        with self._lock:
            if not self.predictions:
                return {}
                
            recent_predictions = list(self.predictions)[-1000:]  # Last 1000 predictions
            
            # Calculate performance metrics
            high_risk_threshold = 0.5
            medium_risk_threshold = 0.25
            
            high_risk = sum(1 for p in recent_predictions if p.prediction > high_risk_threshold)
            medium_risk = sum(1 for p in recent_predictions if medium_risk_threshold < p.prediction <= high_risk_threshold)
            low_risk = sum(1 for p in recent_predictions if p.prediction <= medium_risk_threshold)
            
            total = len(recent_predictions)
            
            return {
                "total_predictions": total,
                "risk_distribution": {
                    "high_risk": {"count": high_risk, "percentage": round(high_risk/total*100, 1) if total > 0 else 0},
                    "medium_risk": {"count": medium_risk, "percentage": round(medium_risk/total*100, 1) if total > 0 else 0},
                    "low_risk": {"count": low_risk, "percentage": round(low_risk/total*100, 1) if total > 0 else 0}
                },
                "avg_prediction_time": round(sum(p.prediction_time for p in recent_predictions) / total, 4) if total > 0 else 0,
                "model_versions": list(set(p.model_version for p in recent_predictions))
            }
    
    def _get_uptime_hours(self) -> float:
        """Calculate system uptime in hours."""
        # Simple uptime calculation - in production, use actual system uptime
        if hasattr(self, '_start_time'):
            return (time.time() - self._start_time) / 3600
        return 0.0

# Global metrics collector instance
_metrics_collector = MetricsCollector()

def log_prediction_request(user: str, input_data: Dict[str, Any], prediction: float, 
                          decision: str, prediction_time: float, model_version: str = "unknown") -> str:
    """Log a prediction request."""
    request_id = f"{user}_{int(time.time())}_{hash(str(input_data)) % 1000}"
    
    metric = PredictionMetric(
        timestamp=datetime.now(),
        user=user,
        model_version=model_version,
        prediction_time=prediction_time,
        input_features=input_data,
        prediction=prediction,
        decision=decision,
        request_id=request_id
    )
    
    _metrics_collector.log_prediction(metric)
    return request_id

def log_error(error_type: str, user: str, details: str):
    """Log an error."""
    _metrics_collector.log_error(error_type, user, details)

def get_metrics(window_minutes: int = 60) -> Dict[str, Any]:
    """Get current metrics summary."""
    return _metrics_collector.get_metrics_summary(window_minutes)

def get_prediction_distribution(bins: int = 10) -> Dict[str, List[float]]:
    """Get prediction distribution."""
    return _metrics_collector.get_prediction_distribution(bins)

def get_model_performance() -> Dict[str, Any]:
    """Get model performance metrics."""
    return _metrics_collector.get_model_performance()

class HealthChecker:
    """Health checker with configurable checks."""
    
    def __init__(self):
        self.checks = {}
        
    def register_check(self, name: str, check_func):
        """Register a health check function."""
        self.checks[name] = check_func
        
    def run_health_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_health = True
        
        for name, check_func in self.checks.items():
            try:
                result = check_func()
                results[name] = {
                    "status": "healthy" if result else "unhealthy",
                    "details": result if isinstance(result, dict) else {}
                }
                if not result:
                    overall_health = False
            except Exception as e:
                results[name] = {
                    "status": "error",
                    "details": str(e)
                }
                overall_health = False
        
        return {
            "overall_status": "healthy" if overall_health else "unhealthy",
            "timestamp": datetime.now().isoformat(),
            "checks": results
        }

# Global health checker
health_checker = HealthChecker()

def setup_default_health_checks(model=None):
    """Setup default health checks."""
    
    def model_health():
        return model is not None
    
    def metrics_health():
        return len(_metrics_collector.predictions) >= 0
    
    def memory_health():
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent < 90  # Alert if memory usage > 90%
        except ImportError:
            return True  # Skip if psutil not available
    
    health_checker.register_check("model", model_health)
    health_checker.register_check("metrics", metrics_health)
    health_checker.register_check("memory", memory_health)

# Initialize metrics collector start time
_metrics_collector._start_time = time.time()