"""
Circuit discovery module.

Handles circuit representation, discovery, similarity computation, and metrics.
"""

# Will be populated when modules are migrated
# from .circuit import SparseFeatureCircuit
# from .discovery import CircuitDiscoverer
from .similarity import CircuitSimilarity, equalize_circuits, compute_circuit_similarity
# from .metrics import CircuitMetrics

__all__ = ['CircuitSimilarity', 'equalize_circuits', 'compute_circuit_similarity']

