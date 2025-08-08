import os
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod
import json
from pathlib import Path

class SecretsManager(ABC):
    """Abstract base class for secrets management."""
    
    @abstractmethod
    def get_secret(self, key: str) -> Optional[str]:
        pass
    
    @abstractmethod
    def set_secret(self, key: str, value: str) -> bool:
        pass

class EnvironmentSecretsManager(SecretsManager):
    """Secrets manager that uses environment variables."""
    
    def get_secret(self, key: str) -> Optional[str]:
        return os.getenv(key)
    
    def set_secret(self, key: str, value: str) -> bool:
        os.environ[key] = value
        return True

class FileSecretsManager(SecretsManager):
    """Secrets manager that uses encrypted local files."""
    
    def __init__(self, secrets_file: str = "secrets.json"):
        self.secrets_file = Path(secrets_file)
        self._secrets_cache = {}
        self._load_secrets()
    
    def _load_secrets(self):
        """Load secrets from file."""
        if self.secrets_file.exists():
            try:
                with open(self.secrets_file, 'r') as f:
                    self._secrets_cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._secrets_cache = {}
    
    def _save_secrets(self):
        """Save secrets to file."""
        try:
            os.makedirs(self.secrets_file.parent, exist_ok=True)
            with open(self.secrets_file, 'w') as f:
                json.dump(self._secrets_cache, f, indent=2)
            # Set restrictive permissions
            os.chmod(self.secrets_file, 0o600)
            return True
        except IOError:
            return False
    
    def get_secret(self, key: str) -> Optional[str]:
        return self._secrets_cache.get(key)
    
    def set_secret(self, key: str, value: str) -> bool:
        self._secrets_cache[key] = value
        return self._save_secrets()

class AzureKeyVaultManager(SecretsManager):
    """Secrets manager for Azure Key Vault (placeholder implementation)."""
    
    def __init__(self, vault_url: str):
        self.vault_url = vault_url
        # In production, initialize Azure Key Vault client here
        
    def get_secret(self, key: str) -> Optional[str]:
        # Placeholder - implement Azure Key Vault integration
        # from azure.keyvault.secrets import SecretClient
        # from azure.identity import DefaultAzureCredential
        # 
        # credential = DefaultAzureCredential()
        # client = SecretClient(vault_url=self.vault_url, credential=credential)
        # try:
        #     retrieved_secret = client.get_secret(key)
        #     return retrieved_secret.value
        # except Exception:
        #     return None
        return os.getenv(key)  # Fallback to env vars for now
    
    def set_secret(self, key: str, value: str) -> bool:
        # Placeholder - implement Azure Key Vault integration
        return False

def get_secrets_manager() -> SecretsManager:
    """Factory function to get the appropriate secrets manager."""
    manager_type = os.getenv("SECRETS_MANAGER", "environment").lower()
    
    if manager_type == "file":
        secrets_file = os.getenv("SECRETS_FILE", "config/secrets.json")
        return FileSecretsManager(secrets_file)
    elif manager_type == "azure":
        vault_url = os.getenv("AZURE_KEY_VAULT_URL")
        if vault_url:
            return AzureKeyVaultManager(vault_url)
        else:
            # Fallback to environment if Azure not configured
            return EnvironmentSecretsManager()
    else:
        return EnvironmentSecretsManager()

# Global secrets manager instance
secrets_manager = get_secrets_manager()

def get_secret(key: str, default: Optional[str] = None) -> str:
    """Get a secret value."""
    value = secrets_manager.get_secret(key)
    if value is None and default is not None:
        return default
    if value is None:
        raise ValueError(f"Secret '{key}' not found and no default provided")
    return value

def get_database_url() -> str:
    """Get database URL from secrets."""
    return get_secret("DATABASE_URL", "sqlite:///./credit_gam.db")

def get_mlflow_tracking_uri() -> str:
    """Get MLflow tracking URI from secrets."""
    return get_secret("MLFLOW_TRACKING_URI", "http://mlflow:5000")

def get_jwt_secret_key() -> str:
    """Get JWT secret key from secrets."""
    return get_secret("JWT_SECRET_KEY", "development-key-change-in-production")

def get_model_uri() -> str:
    """Get model URI from secrets."""
    return get_secret("MODEL_URI", "models:/credit_gam/Production")