# hybrid_cluster/constants.py
"""Constants for hybrid cluster generation

This module contains all magic numbers and constants used in the hybrid cluster generation
process according to TR 38.901 standard. Constants are organized by category for clarity.
"""

import tensorflow as tf
import numpy as np

# =============================================================================
# TR 38.901 STANDARD CONSTANTS
# =============================================================================

# Ray offset angles in degrees (TR 38.901 standard)
# These are the fixed ray offset angles used for intra-cluster ray generation
RAY_OFFSETS_DEG = tf.constant([
    0.0447, -0.0447,  0.1413, -0.1413,  0.2492, -0.2492,
    0.3715, -0.3715,  0.5129, -0.5129,  0.6797, -0.6797,
    0.8844, -0.8844,  1.1481, -1.1481,  1.5195, -1.5195,
    2.1551, -2.1551
], dtype=tf.float32)

# =============================================================================
# PHYSICAL CONSTANTS
# =============================================================================

# Speed of light in m/s
SPEED_OF_LIGHT = 3e8

# Mathematical constants
PI = np.pi
DEG_TO_RAD = PI / 180.0
RAD_TO_DEG = 180.0 / PI

# =============================================================================
# VALIDATION THRESHOLDS
# =============================================================================

# Power threshold for cluster filtering (dB)
# Clusters with power below this threshold are considered invalid
POWER_THRESHOLD_DB = -25.0

# Invalid delay value (seconds)
# Used to mark invalid ray-tracing paths
INVALID_DELAY = -1.0
