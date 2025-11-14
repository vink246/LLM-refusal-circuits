"""
Models module.

Handles model loading, inference, and activation collection.
"""

from .model_wrapper import ModelWrapper, ModelConfig, load_model_with_hooks
from .inference_pipeline import InferencePipeline, detect_refusal_from_output

__all__ = [
    'ModelWrapper',
    'ModelConfig',
    'load_model_with_hooks',
    'InferencePipeline',
    'detect_refusal_from_output'
]
