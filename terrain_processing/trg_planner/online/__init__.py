"""Online terrain-processing utilities.

ElevationGridBuilder maintains a full-field elevation representation. Related
modules provide extent/boundary geometry, reference Python graph integration,
point-cloud I/O and optional visualization. Core geometry is ROS-independent;
the ROS 2 wrappers are provided in the module's ros/ directory.
"""

from .elevation_builder import ElevationConfig, ElevationGridBuilder

__all__ = ["ElevationConfig", "ElevationGridBuilder"]
