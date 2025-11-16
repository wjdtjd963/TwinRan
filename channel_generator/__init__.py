# channel_generator/__init__.py
"""
Channel Generator Package

This package provides hybrid cluster generation functionality for wireless channel modeling
based on TR 38.901 standard. The main entry point is HybridClusterGenerator which orchestrates
the entire pipeline from ray-tracing data to final channel coefficients.

Main Components:
- HybridClusterGenerator: Main orchestrator for the complete pipeline
- hybrid_cluster: Source package containing modular components
"""

from .HybridClusterGenerator import HybridClusterGenerator

# Import commonly used constants and configurations
from .hybrid_cluster.constants import (
    RAY_OFFSETS_DEG, POWER_THRESHOLD_DB, INVALID_DELAY,
    SPEED_OF_LIGHT, PI, DEG_TO_RAD, RAD_TO_DEG
)
from .hybrid_cluster.exceptions import (
    HybridClusterError, ValidationError, ConfigurationError,
    RayTracingError, ClusterGenerationError, RayGenerationError,
    ChannelSynthesisError, DataFormatError, ParameterError
)

__all__ = [
    # Main entry point
    "HybridClusterGenerator",
    
    # Common constants
    "RAY_OFFSETS_DEG", "POWER_THRESHOLD_DB", "INVALID_DELAY",
    "SPEED_OF_LIGHT", "PI", "DEG_TO_RAD", "RAD_TO_DEG",
    
    # Exceptions
    "HybridClusterError", "ValidationError", "ConfigurationError",
    "RayTracingError", "ClusterGenerationError", "RayGenerationError",
    "ChannelSynthesisError", "DataFormatError", "ParameterError",
]

__version__ = "1.0.0"
__author__ = "Y-Twin Team"
__description__ = "Hybrid cluster generation for wireless channel modeling"
