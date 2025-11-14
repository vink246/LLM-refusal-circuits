"""
Circuit discovery module.

Handles circuit representation, discovery, similarity computation, and metrics.
"""

from .circuit import SparseFeatureCircuit
from .discovery import CircuitDiscoverer
from .similarity import CircuitSimilarity, equalize_circuits, compute_circuit_similarity
# from .metrics import CircuitMetrics

__all__ = [
    'SparseFeatureCircuit',
    'CircuitDiscoverer',
    'CircuitSimilarity',
    'equalize_circuits',
    'compute_circuit_similarity'
]

