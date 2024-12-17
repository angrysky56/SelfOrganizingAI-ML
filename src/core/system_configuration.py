# Adaptive System Configuration Manager
# Implements dynamic environment detection and protocol adaptation

from dataclasses import dataclass
from typing import Optional, Dict
from pathlib import Path
import logging
import ssl
import json

@dataclass
class EnvironmentContext:
    """
    Environment-aware configuration parameters.
    
    Attributes:
        env_type: Development/Production identifier
        security_level: Required security constraints
        resource_paths: System resource locations
        runtime_flags: Environmental feature flags
    """
    env_type: str
    security_level: str
    resource_paths: Dict[str, Path]
    runtime_flags: Dict[str, bool]

class SystemConfiguration:
    """
    Manages system configuration and environment adaptation.
    
    Core Responsibilities:
    1. Environment detection and configuration
    2. Security protocol selection
    3. Resource path management
    4. Runtime optimization
    """
    
    def __init__(self, base_path: Path):
        self.logger = logging.getLogger(__name__)
        self.base_path = base_path
        self._env_context = self._initialize_environment()
        
    def _initialize_environment(self) -> EnvironmentContext:
        """
        Initialize environment-specific configuration.
        
        Returns:
            EnvironmentContext: Configured environment parameters
        """
        # Detect environment type
        is_development = self._is_development_environment()
        
        # Configure resource paths
        resource_paths = {
            'config': self.base_path / 'config',
            'data': self.base_path / 'data',
            'models': self.base_path / 'models',
            'experiments': self.base_path / 'experiments'
        }
        
        # Set runtime flags
        runtime_flags = {
            'debug_mode': is_development,
            'verify_ssl': not is_development,
            'enable_monitoring': True,
            'allow_hot_reload': is_development
        }
        
        return EnvironmentContext(
            env_type='development' if is_development else 'production',
            security_level='permissive' if is_development else 'strict',
            resource_paths=resource_paths,
            runtime_flags=runtime_flags
        )
    
    def _is_development_environment(self) -> bool:
        """
        Detect development environment indicators.
        
        Returns:
            bool: True if development environment detected
        """
        dev_indicators = [
            self.base_path.name.lower().endswith('-development'),
            (self.base_path / '.git').exists(),
            (self.base_path / 'tests').exists()
        ]
        return any(dev_indicators)
    
    def get_ssl_config(self) -> Dict[str, str]:
        """
        Generate environment-appropriate SSL configuration.
        
        Returns:
            Dict[str, str]: SSL configuration parameters
        """
        is_dev = self._env_context.env_type == 'development'
        
        return {
            'verify_mode': 'none' if is_dev else 'required',
            'cert_dir': str(self._env_context.resource_paths['config'] / 'ssl'),
            'protocol_version': 'TLSv1_2',
            'verify_flags': 'none' if is_dev else 'strict'
        }
    
    def export_config(self) -> str:
        """
        Export current configuration state.
        
        Returns:
            str: JSON configuration representation
        """
        config_state = {
            'environment': self._env_context.env_type,
            'security': {
                'level': self._env_context.security_level,
                'ssl': self.get_ssl_config()
            },
            'paths': {k: str(v) for k, v in self._env_context.resource_paths.items()},
            'flags': self._env_context.runtime_flags
        }
        return json.dumps(config_state, indent=2)
    
    def validate_paths(self) -> bool:
        """
        Validate required system paths.
        
        Returns:
            bool: True if all required paths exist
        """
        try:
            for path in self._env_context.resource_paths.values():
                if not path.exists():
                    path.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"Created missing directory: {path}")
            return True
        except Exception as e:
            self.logger.error(f"Path validation failed: {str(e)}")
            return False