#!/usr/bin/env python3
"""Record one operator-driven path and replay it in either direction."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry, Path as PathMsg
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger


class RecordedPathManager(Node):
    def __init__(self) -> None:
        super().__init__("recorded_path_manager")
        self._load_params()

        self._recording = False
        self._poses: list[PoseStamped] = []
        self._latest_odom: Optional[Odometry] = None
        self._last_record_xy: Optional[tuple[float, float]] = None
        self._last_use_direction = "none"
        self._last_processed_count = 0

        self._sub_odom = self.create_subscription(
            Odometry, self.odom_topic, self._on_odom, 20
        )
        self._pub_path = self.create_publisher(PathMsg, self.output_path_topic, 10)
        self._pub_preview = self.create_publisher(PathMsg, self.preview_path_topic, 10)
        self._pub_status = self.create_publisher(String, self.status_topic, 10)

        self._srv_start = self.create_service(
            Trigger, self.start_recording_service, self._on_start_recording
        )
        self._srv_stop = self.create_service(
            Trigger, self.stop_recording_service, self._on_stop_recording
        )
        self._srv_use = self.create_service(
            Trigger, self.use_recorded_path_service, self._on_use_recorded_path
        )
        self._timer = self.create_timer(1.0, self._publish_status)

        self.get_logger().info(
            "recorded_path_manager ready: "
            f"odom={self.odom_topic}, output={self.output_path_topic}, "
            f"start={self.start_recording_service}, stop={self.stop_recording_service}, "
            f"use={self.use_recorded_path_service}"
        )

    def _load_params(self) -> None:
        self.declare_parameter("odom_topic", "/robot_center/odom")
        self.declare_parameter("output_path_topic", "/mission/recorded_path/active")
        self.declare_parameter("preview_path_topic", "/mission/recorded_path/preview")
        self.declare_parameter("status_topic", "/mission/recorded_path/status")
        self.declare_parameter(
            "start_recording_service", "/mission/recorded_path/start_recording"
        )
        self.declare_parameter(
            "stop_recording_service", "/mission/recorded_path/stop_recording"
        )
        self.declare_parameter(
            "use_recorded_path_service", "/mission/recorded_path/use"
        )
        self.declare_parameter("min_record_spacing", 0.08)
        self.declare_parameter("min_path_points", 3)
        self.declare_parameter("smooth_enabled", True)
        self.declare_parameter("smooth_resample_spacing", 0.18)
        self.declare_parameter("smooth_window", 5)
        self.declare_parameter("smooth_passes", 3)
        self.declare_parameter("smooth_max_offset", 0.15)
        self.declare_parameter("smooth_endpoint_keep_distance", 0.35)
        self.declare_parameter("smooth_prune_reversals", True)
        self.declare_parameter("smooth_prune_max_segment", 0.60)
        self.declare_parameter("smooth_prune_reverse_cos", -0.5)
        self.declare_parameter("smooth_prune_loop_radius", 0.30)
        self.declare_parameter("smooth_prune_loop_min_span", 0.45)
        self.declare_parameter("active_start_skip_distance", 0.20)
        self.declare_parameter("save_path", "")

        self.odom_topic = str(self.get_parameter("odom_topic").value)
        self.output_path_topic = str(self.get_parameter("output_path_topic").value)
        self.preview_path_topic = str(self.get_parameter("preview_path_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)
        self.start_recording_service = str(
            self.get_parameter("start_recording_service").value
        )
        self.stop_recording_service = str(
            self.get_parameter("stop_recording_service").value
        )
        self.use_recorded_path_service = str(
            self.get_parameter("use_recorded_path_service").value
        )
        self.min_record_spacing = max(
            0.0, float(self.get_parameter("min_record_spacing").value)
        )
        self.min_path_points = max(2, int(self.get_parameter("min_path_points").value))
        self.smooth_enabled = bool(self.get_parameter("smooth_enabled").value)
        self.smooth_resample_spacing = max(
            0.0, float(self.get_parameter("smooth_resample_spacing").value)
        )
        self.smooth_window = max(3, int(self.get_parameter("smooth_window").value))
        if self.smooth_window % 2 == 0:
            self.smooth_window += 1
        self.smooth_passes = max(0, int(self.get_parameter("smooth_passes").value))
        self.smooth_max_offset = max(
            0.0, float(self.get_parameter("smooth_max_offset").value)
        )
        self.smooth_endpoint_keep_distance = max(
            0.0, float(self.get_parameter("smooth_endpoint_keep_distance").value)
        )
        self.smooth_prune_reversals = bool(
            self.get_parameter("smooth_prune_reversals").value
        )
        self.smooth_prune_max_segment = max(
            0.0, float(self.get_parameter("smooth_prune_max_segment").value)
        )
        self.smooth_prune_reverse_cos = float(
            self.get_parameter("smooth_prune_reverse_cos").value
        )
        self.smooth_prune_loop_radius = max(
            0.0, float(self.get_parameter("smooth_prune_loop_radius").value)
        )
        self.smooth_prune_loop_min_span = max(
            0.0, float(self.get_parameter("smooth_prune_loop_min_span").value)
        )
        self.active_start_skip_distance = max(
            0.0, float(self.get_parameter("active_start_skip_distance").value)
        )
        self.save_path = str(self.get_parameter("save_path").value).strip()

    def _on_odom(self, msg: Odometry) -> None:
        self._latest_odom = msg
        if not self._recording:
            return
        pose = PoseStamped()
        pose.header = msg.header
        pose.header.frame_id = msg.header.frame_id or "map"
        pose.pose = msg.pose.pose
        xy = (float(pose.pose.position.x), float(pose.pose.position.y))
        if self._last_record_xy is not None:
            dist = math.hypot(xy[0] - self._last_record_xy[0], xy[1] - self._last_record_xy[1])
            if dist < self.min_record_spacing:
                return
        self._poses.append(pose)
        self._last_record_xy = xy
        self._publish_preview()

    def _on_start_recording(
        self, req: Trigger.Request, resp: Trigger.Response
    ) -> Trigger.Response:
        del req
        self._poses = []
        self._last_record_xy = None
        self._last_use_direction = "none"
        self._last_processed_count = 0
        self._recording = True
        if self._latest_odom is not None:
            self._on_odom(self._latest_odom)
        resp.success = True
        resp.message = "recording started; previous recorded path cleared"
        self.get_logger().warn(resp.message)
        self._publish_status()
        return resp

    def _on_stop_recording(
        self, req: Trigger.Request, resp: Trigger.Response
    ) -> Trigger.Response:
        del req
        was_recording = self._recording
        self._recording = False
        count = len(self._poses)
        ok = count >= self.min_path_points
        processed_count = 0
        if ok:
            processed_count = len(self._processed_poses(self._poses))
            self._last_processed_count = processed_count
            self._save_recorded_path()
            self._publish_preview(processed=True)
        resp.success = ok
        if ok:
            resp.message = (
                f"recording stopped; recorded {count} raw path points, "
                f"{processed_count} smoothed points"
            )
        elif was_recording:
            resp.message = (
                f"recording stopped but path is too short: {count} points "
                f"< {self.min_path_points}"
            )
        else:
            resp.message = (
                f"not recording; recorded path has {count} points "
                f"({'usable' if ok else 'not usable'})"
            )
        if ok:
            self.get_logger().warn(resp.message)
        else:
            self.get_logger().error(resp.message)
        self._publish_status()
        return resp

    def _on_use_recorded_path(
        self, req: Trigger.Request, resp: Trigger.Response
    ) -> Trigger.Response:
        del req
        if len(self._poses) < self.min_path_points:
            resp.success = False
            resp.message = (
                f"recorded path unavailable: {len(self._poses)} points "
                f"< {self.min_path_points}"
            )
            self.get_logger().error(resp.message)
            return resp
        if self._latest_odom is None:
            resp.success = False
            resp.message = "recorded path unavailable: no odometry yet"
            self.get_logger().error(resp.message)
            return resp

        robot = self._latest_odom.pose.pose.position
        first = self._poses[0].pose.position
        last = self._poses[-1].pose.position
        d_first = math.hypot(robot.x - first.x, robot.y - first.y)
        d_last = math.hypot(robot.x - last.x, robot.y - last.y)

        if d_first <= d_last:
            ordered = self._poses
            near = "start"
            far = "end"
            far_pose = last
        else:
            ordered = list(reversed(self._poses))
            near = "end"
            far = "start"
            far_pose = first

        full_msg = PathMsg()
        full_msg.header.stamp = self.get_clock().now().to_msg()
        full_msg.header.frame_id = ordered[0].header.frame_id or "map"
        full_msg.poses = self._processed_poses(
            ordered, full_msg.header.stamp, full_msg.header.frame_id
        )
        active_msg = PathMsg()
        active_msg.header = full_msg.header
        active_msg.poses = self._skip_initial_distance(
            full_msg.poses,
            self.active_start_skip_distance,
            full_msg.header.stamp,
            full_msg.header.frame_id,
        )
        self._pub_path.publish(active_msg)
        self._pub_preview.publish(active_msg)
        self._last_processed_count = len(active_msg.poses)
        self._last_use_direction = f"{near}_to_{far}"
        resp.success = True
        resp.message = (
            f"recorded path published {self._last_use_direction}; "
            f"robot distances start={d_first:.2f}m end={d_last:.2f}m; "
            f"target={far} ({far_pose.x:.2f}, {far_pose.y:.2f}); "
            f"raw_points={len(ordered)} smoothed_points={len(full_msg.poses)} "
            f"active_points={len(active_msg.poses)} "
            f"start_skip={self.active_start_skip_distance:.2f}m "
            f"smoothing={str(self.smooth_enabled).lower()}"
        )
        self.get_logger().warn(resp.message)
        self._publish_status()
        return resp

    def _publish_preview(self, processed: bool = False) -> None:
        if not self._poses:
            return
        msg = PathMsg()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._poses[0].header.frame_id or "map"
        if processed:
            msg.poses = self._processed_poses(self._poses, msg.header.stamp, msg.header.frame_id)
        else:
            msg.poses = [
                self._restamp_pose(pose, msg.header.stamp, msg.header.frame_id)
                for pose in self._poses
            ]
        self._pub_preview.publish(msg)

    def _publish_status(self) -> None:
        msg = String()
        msg.data = (
            f"recording={str(self._recording).lower()}; "
            f"points={len(self._poses)}; "
            f"smoothed_points={self._last_processed_count}; "
            f"smoothing={str(self.smooth_enabled).lower()}; "
            f"usable={str(len(self._poses) >= self.min_path_points).lower()}; "
            f"last_use={self._last_use_direction}"
        )
        self._pub_status.publish(msg)

    def _save_recorded_path(self) -> None:
        if not self.save_path:
            return
        path = Path(self.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = self.get_clock().now().to_msg()
        frame_id = self._poses[0].header.frame_id if self._poses else "map"
        smoothed = self._processed_poses(self._poses, stamp, frame_id)
        payload = {
            "frame_id": frame_id,
            "smoothing": {
                "enabled": self.smooth_enabled,
                "resample_spacing": self.smooth_resample_spacing,
                "window": self.smooth_window,
                "passes": self.smooth_passes,
                "max_offset": self.smooth_max_offset,
                "endpoint_keep_distance": self.smooth_endpoint_keep_distance,
                "prune_reversals": self.smooth_prune_reversals,
                "prune_max_segment": self.smooth_prune_max_segment,
                "prune_reverse_cos": self.smooth_prune_reverse_cos,
                "prune_loop_radius": self.smooth_prune_loop_radius,
                "prune_loop_min_span": self.smooth_prune_loop_min_span,
                "active_start_skip_distance": self.active_start_skip_distance,
            },
            "points": [
                {
                    "x": float(pose.pose.position.x),
                    "y": float(pose.pose.position.y),
                    "z": float(pose.pose.position.z),
                    "qx": float(pose.pose.orientation.x),
                    "qy": float(pose.pose.orientation.y),
                    "qz": float(pose.pose.orientation.z),
                    "qw": float(pose.pose.orientation.w),
                }
                for pose in self._poses
            ],
            "smoothed_points": [
                {
                    "x": float(pose.pose.position.x),
                    "y": float(pose.pose.position.y),
                    "z": float(pose.pose.position.z),
                    "qx": float(pose.pose.orientation.x),
                    "qy": float(pose.pose.orientation.y),
                    "qz": float(pose.pose.orientation.z),
                    "qw": float(pose.pose.orientation.w),
                }
                for pose in smoothed
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _processed_poses(
        self, poses: list[PoseStamped], stamp=None, frame_id: Optional[str] = None
    ) -> list[PoseStamped]:
        if not poses:
            return []
        if stamp is None:
            stamp = self.get_clock().now().to_msg()
        frame_id = frame_id or poses[0].header.frame_id or "map"
        if not self.smooth_enabled or len(poses) < 3:
            return [self._restamp_pose(pose, stamp, frame_id) for pose in poses]

        points = self._poses_to_points(poses)
        points = self._dedupe_points(points, min_spacing=0.02)
        if len(points) < 3:
            return self._points_to_poses(points, stamp, frame_id)
        if self.smooth_prune_reversals:
            points = self._prune_reversals(points)
        resampled = self._resample_points(points, self.smooth_resample_spacing)
        smoothed = self._smooth_points(resampled)
        return self._points_to_poses(smoothed, stamp, frame_id)

    @staticmethod
    def _poses_to_points(poses: list[PoseStamped]) -> list[tuple[float, float, float]]:
        return [
            (
                float(pose.pose.position.x),
                float(pose.pose.position.y),
                float(pose.pose.position.z),
            )
            for pose in poses
        ]

    @staticmethod
    def _dedupe_points(
        points: list[tuple[float, float, float]], min_spacing: float
    ) -> list[tuple[float, float, float]]:
        if not points:
            return []
        out = [points[0]]
        for point in points[1:]:
            if math.hypot(point[0] - out[-1][0], point[1] - out[-1][1]) >= min_spacing:
                out.append(point)
        if out[-1] != points[-1]:
            out.append(points[-1])
        return out

    def _prune_reversals(
        self, points: list[tuple[float, float, float]]
    ) -> list[tuple[float, float, float]]:
        current = self._prune_near_loops(points)
        current = self._prune_short_reversals(current)
        return current

    def _prune_near_loops(
        self, points: list[tuple[float, float, float]]
    ) -> list[tuple[float, float, float]]:
        if len(points) < 4 or self.smooth_prune_loop_radius <= 0.0:
            return points

        current = points
        for _ in range(4):
            cumulative = self._cumulative_lengths(current)
            protected = self._endpoint_protection_mask(cumulative, keep_distance=0.0)
            changed = False
            out: list[tuple[tuple[float, float, float], float, bool]] = []

            for point, distance, is_protected in zip(current, cumulative, protected):
                candidate = None
                if not is_protected and len(out) >= 3:
                    for j in range(len(out) - 2, -1, -1):
                        ref, ref_distance, ref_protected = out[j]
                        if ref_protected:
                            continue
                        if distance - ref_distance < self.smooth_prune_loop_min_span:
                            continue
                        if any(item[2] for item in out[j + 1 :]):
                            continue
                        if (
                            math.hypot(point[0] - ref[0], point[1] - ref[1])
                            <= self.smooth_prune_loop_radius
                        ):
                            candidate = j
                            break

                if candidate is not None:
                    out = out[: candidate + 1]
                    changed = True
                    continue

                out.append((point, distance, is_protected))

            current = [item[0] for item in out]
            if not changed or len(current) < 4:
                break
        return current

    def _prune_short_reversals(
        self, points: list[tuple[float, float, float]]
    ) -> list[tuple[float, float, float]]:
        if len(points) < 3 or self.smooth_prune_max_segment <= 0.0:
            return points
        current = points
        for _ in range(3):
            cumulative = self._cumulative_lengths(current)
            protected = self._endpoint_protection_mask(cumulative, keep_distance=0.0)
            changed = False
            pruned = [current[0]]
            i = 1
            while i < len(current) - 1:
                if protected[i]:
                    pruned.append(current[i])
                    i += 1
                    continue
                prev = pruned[-1]
                curr = current[i]
                nxt = current[i + 1]
                v1 = (curr[0] - prev[0], curr[1] - prev[1])
                v2 = (nxt[0] - curr[0], nxt[1] - curr[1])
                l1 = math.hypot(v1[0], v1[1])
                l2 = math.hypot(v2[0], v2[1])
                reverse = False
                if (
                    1e-6 < l1 <= self.smooth_prune_max_segment
                    and 1e-6 < l2 <= self.smooth_prune_max_segment
                ):
                    cos_angle = (v1[0] * v2[0] + v1[1] * v2[1]) / (l1 * l2)
                    reverse = cos_angle <= self.smooth_prune_reverse_cos
                if reverse:
                    changed = True
                    i += 1
                    continue
                pruned.append(curr)
                i += 1
            pruned.append(current[-1])
            current = pruned
            if not changed or len(current) < 3:
                break
        return current

    @staticmethod
    def _cumulative_lengths(
        points: list[tuple[float, float, float]]
    ) -> list[float]:
        if not points:
            return []
        cumulative = [0.0]
        for a, b in zip(points, points[1:]):
            cumulative.append(cumulative[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        return cumulative

    def _endpoint_protection_mask(
        self, cumulative: list[float], keep_distance: Optional[float] = None
    ) -> list[bool]:
        if not cumulative:
            return []
        keep = self.smooth_endpoint_keep_distance if keep_distance is None else keep_distance
        protected = [False] * len(cumulative)
        protected[0] = True
        protected[-1] = True
        if keep <= 0.0:
            return protected
        total = cumulative[-1]
        return [
            is_protected or d <= keep or total - d <= keep
            for d, is_protected in zip(cumulative, protected)
        ]

    def _skip_initial_distance(
        self,
        poses: list[PoseStamped],
        distance: float,
        stamp,
        frame_id: str,
    ) -> list[PoseStamped]:
        if len(poses) < 2 or distance <= 0.0:
            return poses
        points = self._poses_to_points(poses)
        cumulative = self._cumulative_lengths(points)
        total = cumulative[-1] if cumulative else 0.0
        if total <= max(distance + 1e-6, 0.30):
            return poses
        skip = min(distance, max(0.0, total - 0.30))
        seg = 0
        while seg + 1 < len(cumulative) and cumulative[seg + 1] < skip:
            seg += 1
        if seg + 1 >= len(points):
            return poses
        a = points[seg]
        b = points[seg + 1]
        length = max(1e-9, cumulative[seg + 1] - cumulative[seg])
        t = (skip - cumulative[seg]) / length
        first = (
            a[0] + (b[0] - a[0]) * t,
            a[1] + (b[1] - a[1]) * t,
            a[2] + (b[2] - a[2]) * t,
        )
        active_points = [first] + points[seg + 1 :]
        return self._points_to_poses(active_points, stamp, frame_id)

    @staticmethod
    def _resample_points(
        points: list[tuple[float, float, float]], spacing: float
    ) -> list[tuple[float, float, float]]:
        if len(points) < 2 or spacing <= 0.0:
            return points
        cumulative = [0.0]
        for a, b in zip(points, points[1:]):
            cumulative.append(cumulative[-1] + math.hypot(b[0] - a[0], b[1] - a[1]))
        total = cumulative[-1]
        if total <= 1e-6:
            return [points[0], points[-1]]

        targets = []
        n_steps = max(1, int(math.floor(total / spacing)))
        for i in range(n_steps + 1):
            targets.append(min(total, i * spacing))
        if targets[-1] < total:
            targets.append(total)

        out = []
        seg = 0
        for target in targets:
            while seg + 1 < len(cumulative) and cumulative[seg + 1] < target:
                seg += 1
            if seg + 1 >= len(points):
                out.append(points[-1])
                continue
            a = points[seg]
            b = points[seg + 1]
            length = max(1e-9, cumulative[seg + 1] - cumulative[seg])
            t = (target - cumulative[seg]) / length
            out.append(
                (
                    a[0] + (b[0] - a[0]) * t,
                    a[1] + (b[1] - a[1]) * t,
                    a[2] + (b[2] - a[2]) * t,
                )
            )
        return out

    def _smooth_points(
        self, points: list[tuple[float, float, float]]
    ) -> list[tuple[float, float, float]]:
        if len(points) < 3 or self.smooth_passes <= 0:
            return points
        baseline = points
        current = points
        half_window = self.smooth_window // 2
        for _ in range(self.smooth_passes):
            cumulative = self._cumulative_lengths(current)
            protected = self._endpoint_protection_mask(cumulative)
            nxt = [current[0]]
            for i in range(1, len(current) - 1):
                if protected[i]:
                    nxt.append(baseline[i])
                    continue
                lo = max(0, i - half_window)
                hi = min(len(current), i + half_window + 1)
                count = hi - lo
                proposed = (
                    sum(p[0] for p in current[lo:hi]) / count,
                    sum(p[1] for p in current[lo:hi]) / count,
                    sum(p[2] for p in current[lo:hi]) / count,
                )
                nxt.append(self._limit_offset(proposed, baseline[i], self.smooth_max_offset))
            nxt.append(current[-1])
            current = nxt
        return current

    @staticmethod
    def _limit_offset(
        point: tuple[float, float, float],
        anchor: tuple[float, float, float],
        max_offset: float,
    ) -> tuple[float, float, float]:
        if max_offset <= 0.0:
            return anchor
        dx = point[0] - anchor[0]
        dy = point[1] - anchor[1]
        dist = math.hypot(dx, dy)
        if dist <= max_offset or dist <= 1e-9:
            return point
        scale = max_offset / dist
        return (anchor[0] + dx * scale, anchor[1] + dy * scale, point[2])

    def _points_to_poses(
        self, points: list[tuple[float, float, float]], stamp, frame_id: str
    ) -> list[PoseStamped]:
        poses = []
        for i, point in enumerate(points):
            pose = PoseStamped()
            pose.header.stamp = stamp
            pose.header.frame_id = frame_id
            pose.pose.position.x = point[0]
            pose.pose.position.y = point[1]
            pose.pose.position.z = point[2]
            yaw = self._path_yaw(points, i)
            pose.pose.orientation.z = math.sin(0.5 * yaw)
            pose.pose.orientation.w = math.cos(0.5 * yaw)
            poses.append(pose)
        return poses

    @staticmethod
    def _path_yaw(points: list[tuple[float, float, float]], index: int) -> float:
        if len(points) < 2:
            return 0.0
        if index + 1 < len(points):
            a = points[index]
            b = points[index + 1]
        else:
            a = points[index - 1]
            b = points[index]
        return math.atan2(b[1] - a[1], b[0] - a[0])

    @staticmethod
    def _restamp_pose(pose: PoseStamped, stamp, frame_id: str) -> PoseStamped:
        out = PoseStamped()
        out.header.stamp = stamp
        out.header.frame_id = frame_id
        out.pose = pose.pose
        return out


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RecordedPathManager()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
