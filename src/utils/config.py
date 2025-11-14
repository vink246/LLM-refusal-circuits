"""
Configuration loading utilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dictionary containing configuration
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config: Dict[str, Any], required_fields: list) -> bool:
    """
    Validate that configuration has required fields.
    
    Args:
        config: Configuration dictionary
        required_fields: List of required field paths (e.g., ['model.name', 'data.path'])
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    missing_fields = []
    
    for field_path in required_fields:
        keys = field_path.split('.')
        current = config
        
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                missing_fields.append(field_path)
                break
            current = current[key]
    
    if missing_fields:
        raise ValueError(f"Configuration missing required fields: {missing_fields}")
    
    return True

