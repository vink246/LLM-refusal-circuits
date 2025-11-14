"""
Utility functions module.

Provides configuration loading, logging setup, and path management.
"""

from .config import load_config, validate_config
from .logging import setup_logging

# Will be populated when modules are migrated
# from .paths import get_results_dir, get_data_dir

__all__ = ['load_config', 'validate_config', 'setup_logging']

