# hybrid_cluster/exceptions.py
"""Custom exceptions for hybrid cluster generation"""


class HybridClusterError(Exception):
    """Base exception for hybrid cluster generation"""
    pass


class ValidationError(HybridClusterError):
    """Raised when input validation fails"""
    pass


class ConfigurationError(HybridClusterError):
    """Raised when configuration is invalid"""
    pass


class RayTracingError(HybridClusterError):
    """Raised when ray-tracing fails"""
    pass


class ClusterGenerationError(HybridClusterError):
    """Raised when cluster generation fails"""
    pass


class RayGenerationError(HybridClusterError):
    """Raised when ray generation fails"""
    pass


class ChannelSynthesisError(HybridClusterError):
    """Raised when channel synthesis fails"""
    pass


class DataFormatError(HybridClusterError):
    """Raised when data format is invalid"""
    pass


class ParameterError(HybridClusterError):
    """Raised when parameters are invalid"""
    pass
