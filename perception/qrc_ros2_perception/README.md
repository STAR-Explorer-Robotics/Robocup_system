# ROS 2 perception bridge

ROS 2 package qrc_ros2_perception connects sensor images to the sibling [inspection runtime](../inspection/README.md).

## Entry points

| Entry | Role |
| --- | --- |
| qrc-sensing-node | Image acquisition, processing and ROS publication |
| qrc-result-panel | Persistent result display |
| qrc-test-image-publisher | Small image-input helper |

## Source map

- [node.py](qrc_ros2_perception/node.py): sensing node.
- [input_manager.py](qrc_ros2_perception/input_manager.py): input source handling.
- [processing.py](qrc_ros2_perception/processing.py): runtime integration.
- [publishers.py](qrc_ros2_perception/publishers.py): image/result publication.
- [result_panel_node.py](qrc_ros2_perception/result_panel_node.py): result panel.
- [launch/qrc_sensing.launch.py](launch/qrc_sensing.launch.py): module launch.
- [configs/robotdog.example.yaml](configs/robotdog.example.yaml): representative ROS topic configuration.

## Dependencies and scope

Requires ROS 2 Humble, rclpy, sensor/std messages, cv_bridge and the sibling inspection source/runtime dependencies. Package and container metadata are retained for source review; environment compatibility is not certified in this edition.

This bridge provides sensing outputs and visualization. It does not include autonomous arm execution or a complete competition operator application.

See [license information](../../LICENSES.md).

