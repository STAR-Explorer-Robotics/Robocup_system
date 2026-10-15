"""Compatibility adapter for sensing run artifact writing."""

from rescue_inspection.runtime import SensingRunArtifactWriter


class Ros2RunOutputWriter(SensingRunArtifactWriter):
    """Backward-compatible ROS2 name for the inspection runtime artifact writer."""
