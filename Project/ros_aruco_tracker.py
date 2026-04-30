import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
import csv
import math

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose2D


class ArucoPosePublisher(Node):
    def __init__(self):
        super().__init__('aruco_pose_publisher')

        self.pose_pub = self.create_publisher(Pose2D, '/aruco_pose', 10)

        self.url = 'http://10.20.108.207:8080/video'
        self.cap = cv2.VideoCapture(self.url)

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.parameters)

        self.marker_positions = defaultdict(list)
        self.frame_count = 0

        self.target_marker_id = None  # set to an int like 0 if you only want one tag

        self.timer = self.create_timer(0.02, self.detect_and_publish)  # 50 Hz

        self.get_logger().info("Starting ArUco Pose Publisher...")

    def detect_and_publish(self):
        ret, frame = self.cap.read()

        if not ret:
            self.get_logger().warn("Failed to read frame from stream")
            return

        corners, ids, rejected = self.detector.detectMarkers(frame)

        if ids is None:
            if self.frame_count % 100 == 0:
                self.get_logger().info(f"Frame {self.frame_count}: No markers detected")
            self.frame_count += 1
            return

        for i, marker_id in enumerate(ids):
            mid = int(marker_id[0])

            if self.target_marker_id is not None and mid != self.target_marker_id:
                continue

            corner = corners[i][0]

            center_x = float(np.mean(corner[:, 0]))
            center_y = float(np.mean(corner[:, 1]))

            dx = corner[1][0] - corner[0][0]
            dy = corner[1][1] - corner[0][1]

            theta = math.atan2(dy, dx)  # radians

            self.marker_positions[mid].append([
                center_x,
                center_y,
                math.degrees(theta),
                self.frame_count
            ])

            msg = Pose2D()
            msg.x = center_x
            msg.y = center_y
            msg.theta = theta

            self.pose_pub.publish(msg)

            self.get_logger().info(
                f"ID {mid} | x={center_x:.1f}, y={center_y:.1f}, theta={theta:.2f} rad"
            )

            break  # only publish one marker per frame

        self.frame_count += 1

    def save_csv(self):
        if not self.marker_positions:
            print("\nNo markers detected.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aruco_positions_{timestamp}.csv"

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Marker_ID', 'Frame', 'X', 'Y', 'Angle_degrees'])

            for marker_id in sorted(self.marker_positions.keys()):
                for x, y, angle, frame in self.marker_positions[marker_id]:
                    writer.writerow([marker_id, frame, x, y, f"{angle:.2f}"])

        print(f"\nPositions saved to: {filename}")
        print(f"Total frames: {self.frame_count}")
        print(f"Markers detected: {sorted(self.marker_positions.keys())}")

    def cleanup(self):
        self.cap.release()
        self.save_csv()


def main():
    rclpy.init()
    node = ArucoPosePublisher()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        node.cleanup()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()