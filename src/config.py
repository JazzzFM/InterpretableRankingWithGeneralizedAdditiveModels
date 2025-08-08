import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
import re
from dataclasses import dataclass

@dataclass
class MLflowConfig:
    tracking_uri: str
    experiment_name: str
    model_name: str

@dataclass  
class APIConfig:
    host: str
    port: int
    debug: bool
    reload: bool
    workers: Optional[int] = None
    cors_origins: str = ""
    trusted_hosts: str = ""

@dataclass
class SecurityConfig:
    jwt_secret_key: str
    access_token_expire_minutes: int
    secrets_manager: str

@dataclass
class MonitoringConfig:
    log_level: str
    metrics_retention_hours: int
    enable_detailed_logging: bool
    enable_metrics_export: bool = False

@dataclass
class DatabaseConfig:
    url: str

@dataclass
class PerformanceConfig:
    model_cache_size: int = 1
    prediction_timeout_seconds: int = 10
    max_batch_size: int = 50

@dataclass
class FeaturesConfig:
    enable_fairness_evaluation: bool = False
    enable_drift_detection: bool = False
    enable_batch_processing: bool = True
    enable_model_explanation: bool = False

@dataclass
class AlertingConfig:
    slack_webhook: Optional[str] = None
    email_alerts: Optional[str] = None
    alert_thresholds: Optional[Dict[str, float]] = None

class ConfigManager:
    """Centralized configuration management with environment support."""
    
    def __init__(self, environment: Optional[str] = None, config_path: Optional[str] = None):
        """
        Initialize configuration manager.
        
        Args:
            environment: Environment name (development, staging, production)
            config_path: Custom path to config file
        """
        self.environment = environment or os.getenv("ENVIRONMENT", "development")
        self.config_path = config_path or f"configs/{self.environment}.yaml"
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with environment variable substitution."""
        config_file = Path(self.config_path)
        
        if not config_file.exists():
            # Fallback to base config
            config_file = Path("configs/base.yaml")
            
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(config_file, 'r') as f:
            content = f.read()
        
        # Substitute environment variables
        content = self._substitute_env_vars(content)
        
        config = yaml.safe_load(content)
        
        # Override with environment variables if they exist
        config = self._override_with_env_vars(config)
        
        return config
    
    def _substitute_env_vars(self, content: str) -> str:
        """Substitute ${VAR} patterns with environment variables."""
        def replace_env_var(match):
            var_name = match.group(1)
            default_value = match.group(2) if match.group(2) else ""
            return os.getenv(var_name, default_value)
        
        # Pattern: ${VAR_NAME:default_value} or ${VAR_NAME}
        pattern = r'\$\{([^}:]+)(?::([^}]*))?\}'
        return re.sub(pattern, replace_env_var, content)
    
    def _override_with_env_vars(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Override config values with environment variables."""
        env_mapping = {
            'MLFLOW_TRACKING_URI': ['mlflow', 'tracking_uri'],
            'MLFLOW_EXPERIMENT_NAME': ['mlflow', 'experiment_name'],
            'DATABASE_URL': ['database', 'url'],
            'JWT_SECRET_KEY': ['security', 'jwt_secret_key'],
            'API_HOST': ['api', 'host'],
            'API_PORT': ['api', 'port'],
            'LOG_LEVEL': ['monitoring', 'log_level'],
        }
        
        for env_var, path in env_mapping.items():
            value = os.getenv(env_var)
            if value:
                self._set_nested_value(config, path, value)
        
        return config
    
    def _set_nested_value(self, config: Dict[str, Any], path: list, value: Any):
        """Set a nested dictionary value using a path list."""
        current = config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert to appropriate type
        if path[-1] == 'port':
            value = int(value)
        elif path[-1] in ['debug', 'reload', 'enable_detailed_logging', 'enable_metrics_export']:
            value = value.lower() in ('true', '1', 'yes')
        
        current[path[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        keys = key.split('.')
        current = self._config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def get_mlflow_config(self) -> MLflowConfig:
        """Get MLflow configuration."""
        mlflow_config = self._config.get('mlflow', {})
        return MLflowConfig(
            tracking_uri=mlflow_config.get('tracking_uri', 'http://localhost:5000'),
            experiment_name=mlflow_config.get('experiment_name', 'credit-gam'),
            model_name=mlflow_config.get('model_name', 'credit_gam')
        )
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration."""
        api_config = self._config.get('api', {})
        return APIConfig(
            host=api_config.get('host', '0.0.0.0'),
            port=api_config.get('port', 8080),
            debug=api_config.get('debug', False),
            reload=api_config.get('reload', False),
            workers=api_config.get('workers'),
            cors_origins=api_config.get('cors_origins', ''),
            trusted_hosts=api_config.get('trusted_hosts', 'localhost,127.0.0.1')
        )
    
    def get_security_config(self) -> SecurityConfig:
        """Get security configuration."""
        security_config = self._config.get('security', {})
        return SecurityConfig(
            jwt_secret_key=security_config.get('jwt_secret_key', 'dev-key'),
            access_token_expire_minutes=security_config.get('access_token_expire_minutes', 30),
            secrets_manager=security_config.get('secrets_manager', 'environment')
        )
    
    def get_monitoring_config(self) -> MonitoringConfig:
        """Get monitoring configuration."""
        monitoring_config = self._config.get('monitoring', {})
        return MonitoringConfig(
            log_level=monitoring_config.get('log_level', 'INFO'),
            metrics_retention_hours=monitoring_config.get('metrics_retention_hours', 24),
            enable_detailed_logging=monitoring_config.get('enable_detailed_logging', False),
            enable_metrics_export=monitoring_config.get('enable_metrics_export', False)
        )
    
    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration."""
        database_config = self._config.get('database', {})
        return DatabaseConfig(
            url=database_config.get('url', 'sqlite:///./credit_gam.db')
        )
    
    def get_performance_config(self) -> PerformanceConfig:
        """Get performance configuration."""
        performance_config = self._config.get('performance', {})
        return PerformanceConfig(
            model_cache_size=performance_config.get('model_cache_size', 1),
            prediction_timeout_seconds=performance_config.get('prediction_timeout_seconds', 10),
            max_batch_size=performance_config.get('max_batch_size', 50)
        )
    
    def get_features_config(self) -> FeaturesConfig:
        """Get feature flags configuration."""
        features_config = self._config.get('features', {})
        return FeaturesConfig(
            enable_fairness_evaluation=features_config.get('enable_fairness_evaluation', False),
            enable_drift_detection=features_config.get('enable_drift_detection', False),
            enable_batch_processing=features_config.get('enable_batch_processing', True),
            enable_model_explanation=features_config.get('enable_model_explanation', False)
        )
    
    def get_alerting_config(self) -> AlertingConfig:
        """Get alerting configuration."""
        alerting_config = self._config.get('alerting', {})
        return AlertingConfig(
            slack_webhook=alerting_config.get('slack_webhook'),
            email_alerts=alerting_config.get('email_alerts'),
            alert_thresholds=alerting_config.get('alert_thresholds', {})
        )
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment.lower() == 'production'
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment.lower() == 'development'
    
    def get_environment(self) -> str:
        """Get current environment name."""
        return self.environment
    
    def reload(self):
        """Reload configuration from file."""
        self._config = self._load_config()

# Global configuration instance
_config_manager = None

def get_config(environment: Optional[str] = None) -> ConfigManager:
    """Get global configuration manager instance."""
    global _config_manager
    
    if _config_manager is None or (environment and environment != _config_manager.environment):
        _config_manager = ConfigManager(environment)
    
    return _config_manager

def reload_config():
    """Reload the global configuration."""
    global _config_manager
    if _config_manager:
        _config_manager.reload()

# Convenience functions for common configs
def get_mlflow_config() -> MLflowConfig:
    return get_config().get_mlflow_config()

def get_api_config() -> APIConfig:
    return get_config().get_api_config()

def get_security_config() -> SecurityConfig:
    return get_config().get_security_config()

def get_monitoring_config() -> MonitoringConfig:
    return get_config().get_monitoring_config()

def get_features_config() -> FeaturesConfig:
    return get_config().get_features_config()