# Unitree robot interface

The unitree_adapter ROS 2 package for bounded velocity commands and optional Unitree SDK2 sport-mode execution.

## Included functionality

- Velocity/deadband and acceleration limits.
- Command timeout, enable/disable and stop handling.
- An optional bounded body-height service.
- A mock backend and an optional separate-process A2 SDK worker.

## Interfaces

**Inputs:** geometry_msgs/Twist on the configured command topic, plus control services.

**Outputs:** Vendor SDK calls when built/configured for SDK2, adapter state/status and last-command feedback.

See the shared [interface reference](../docs/interfaces.md) before connecting modules.

## Source map

| File | Role |
| --- | --- |
| [src/command_limiter.cpp](src/command_limiter.cpp) | Command limits |
| [src/unitree_adapter_node.cpp](src/unitree_adapter_node.cpp) | ROS adapter and timeout behavior |
| [src/sport_client_backend.cpp](src/sport_client_backend.cpp) | Mock/SDK backend abstraction |
| [src/a2_cmd_vel_bridge.cpp](src/a2_cmd_vel_bridge.cpp) | Optional A2 bridge |
| [config/unitree_adapter.yaml](config/unitree_adapter.yaml) | Mock/backend and control parameters |
| [srv/SetBodyHeight.srv](srv/SetBodyHeight.srv) | Body-height service |
| [LICENSE](LICENSE) | Original Apache-2.0 license |

## Dependencies

ROS 2 Humble and standard message/service packages. Official unitree_sdk2 is an external dependency only when ROBOCUP_UNITREE_USE_SDK2 is enabled.

## Scope and attribution

The ROS package name remains unitree_adapter. This module does not implement collision checking, gait generation or balancing. A separate collision-filter implementation is not included. The imported configuration retains its original enable defaults; no hardware startup is performed by this collection.

The reorganized source is not claimed to be a validated deployment. See [source origins](../docs/source_origins.md) and [license information](../LICENSES.md).

