#!/usr/bin/env python3
"""
Project template.

The PupperInterface class wraps all ROS 2 work: it subscribes to /detections
in a background thread and exposes a small synchronous API that the main loop
can call without dealing with callbacks or executors directly.

API:
    node.get_detections()                -> list[Detection]  (most recent frame)
    node.seconds_since_last_detection()  -> float
    node.set_velocity(linear_x, linear_y, angular_z)   publish Twist to cmd_vel

Fill in the main loop below.
"""

import os
import signal
import threading
import time
from dataclasses import dataclass
from typing import List, Optional

import rclpy
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from rclpy.signals import SignalHandlerOptions
from rclpy.time import Time

from geometry_msgs.msg import Twist
from geometry_msgs.msg import Pose2D

import csv
import math

class PupperInterface(Node):
    """
    ROS 2 node that owns all subscriptions/publishers for the lab.

    Detections arrive on a background spin thread and are cached under a lock
    so that the main thread can grab a consistent snapshot at any time.
    """

    def __init__(self):
        super().__init__('pupper_interface_node')

        self._lock = threading.Lock()

        self._cmd_pub = self.create_publisher(Twist, 'cmd_vel', 10)

        self._latest_pose: Optional[Pose2D] = None
        self._last_detection_time: Optional[float] = None

        self._pose_sub = self.create_subscription(
            Pose2D,
            '/aruco_pose',
            self._pose_callback,
            10
        )

    def _pose_callback(self, msg: Pose2D):
        with self._lock:
            self._latest_pose = msg
            self._last_detection_time = time.time()

    def get_pose(self):
        with self._lock:
            return self._latest_pose

    def seconds_since_last_detection(self):
        with self._lock:
            if self._last_detection_time is None:
                return float('inf')
            return time.time() - self._last_detection_time
    
    def set_velocity(self, linear_x: float = 0.0, linear_y: float = 0.0,
                     angular_z: float = 0.0):
        """Publish a body-frame velocity command to the RL controller.

        linear_x:  forward/back (m/s)
        linear_y:  lateral / strafe (m/s)
        angular_z: yaw rate (rad/s)
        """
        cmd = Twist()
        cmd.linear.x = float(linear_x)
        cmd.linear.y = float(linear_y)
        cmd.angular.z = float(angular_z)
        self._cmd_pub.publish(cmd)

def get_path(input):
    path = []
    with open(input, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            x = float(row[0])
            y = float(row[1])

            path.append([x,y])

    return path

def main():
    # Disable rclpy's default SIGINT handler so Ctrl+C doesn't tear down the
    # context before we get a chance to publish a zero-velocity stop command.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = PupperInterface()

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    stop_requested = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop_requested.set())


    try:
        # ================================================================
        # TODO: control loop.
        #
        # Available:
        #   node.get_detections()                -> list[Detection]
        #   node.seconds_since_last_detection()  -> float
        #   node.set_velocity(linear_x, linear_y, angular_z)
        # ================================================================
        path = get_path('drawn_points.csv')
        target_index = 0
        while rclpy.ok() and not stop_requested.is_set():
            pose = node.get_pose()

            if pose is None or node.seconds_since_last_detection() > 0.5:
                node.set_velocity(0.0, 0.0, 0.0)
                print("No recent ArUco pose. Stopping.")
                time.sleep(0.05)
                continue

            x = pose.x
            y = pose.y
            theta = pose.theta + math.pi/2

            target_x, target_y = path[target_index]

            dx = target_x - x
            dy = target_y - y

            dist = (dx**2 + dy**2)**0.5

            if dist < 30 and target_index < len(path) - 1:
                target_index += 1

            target_angle = math.atan2(dy, dx)
            angle_error = target_angle - theta
            angle_error = math.atan2(math.sin(angle_error), math.cos(angle_error))

            
            print("distance:", dist)
            print("target index:", target_index)

            body_x = math.cos(theta) * dx + math.sin(theta) * dy
            body_y = -math.sin(theta) * dx + math.cos(theta) * dy


            k_forward = 0.2
            k_turn = 0.2

            max_wz = 0.8
            wz = k_turn * angle_error

            if abs(angle_error) < 0.5:
                wz = 0
                print("farts")

            wz = max(-max_wz, min(max_wz, wz))

            if abs(angle_error) < 5:
                vx = k_forward * dist
            else:
                vx = 0.0


            # clamp speeds
            max_speed = 0.4
            vx = max(-max_speed, min(max_speed, vx))
            vy = 0

            node.set_velocity(vx, 0, wz)

            # node.set_velocity(linear_x=0.5, linear_y=0.0, angular_z=0.0)  # move forward at 0.2 m/s
            print(f"vx: {vx}, vy: {vy}, wz: {wz}")
            print(f"angle error: {angle_error}")
            time.sleep(0.1)

    except Exception as e:
        node.get_logger().error(f'Main loop error: {e}')
    finally:
        # Stop the robot. Publish zero velocity a few times with small sleeps
        # so the controller actually receives it before we tear the node down.
        for _ in range(5):
            node.set_velocity(0.0, 0.0, 0.0)
            time.sleep(0.1)
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)


if __name__ == '__main__':
    main()
