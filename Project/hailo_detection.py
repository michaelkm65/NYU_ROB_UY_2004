#!/usr/bin/env python3
"""
ROS2 node for Hailo object detection and tracking
"""

import rclpy
from rclpy.node import Node
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from sensor_msgs.msg import Image, CompressedImage
from std_msgs.msg import String
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import supervision as sv
import numpy as np
import cv2
import queue
import sys
import os
from typing import Dict, List, Tuple
import threading
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import HailoAsyncInference
from fisheye_converter import load_camera_model, fisheye_to_equirectangular


class HailoDetectionNode(Node):
    def __init__(self):
        super().__init__('hailo_detection_node')

        # Declare and get parameters
        self.declare_parameter('model_path', 'yolov5m_wo_spp_60p.hef')
        self.declare_parameter('labels_path', 'coco.txt')
        self.declare_parameter('score_threshold', 0.5)
        self.declare_parameter('use_equirectangular', True)
        self.declare_parameter('camera_params_path', 'camera_params.yaml')
        self.declare_parameter('jpeg_quality', 50)  # Balanced compression for Foxglove
        self.declare_parameter('equirect_scale', 0.5)  # Smaller image for less data
        self.declare_parameter('target_fps', 5.0)  # Limit to 5 FPS to prevent buffer overflow
        self.declare_parameter('publish_equirect', True)  # Publish equirect so Foxglove can switch to it
        self.declare_parameter('annotated_scale', 1.0)  # Full resolution for better viewing

        self.model_path = self.get_parameter('model_path').value
        self.labels_path = self.get_parameter('labels_path').value
        self.score_threshold = self.get_parameter('score_threshold').value
        self.use_equirectangular = self.get_parameter('use_equirectangular').value
        self.camera_params_path = self.get_parameter('camera_params_path').value
        self.jpeg_quality = self.get_parameter('jpeg_quality').value
        self.equirect_scale = self.get_parameter('equirect_scale').value
        self.target_fps = self.get_parameter('target_fps').value
        self.publish_equirect = self.get_parameter('publish_equirect').value
        self.annotated_scale = self.get_parameter('annotated_scale').value
        
        # Frame rate limiting
        self.min_frame_interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0
        self.last_process_time = 0.0
        
        # Log bandwidth optimization settings
        self.get_logger().info(f'Bandwidth Optimization Settings:')
        self.get_logger().info(f'  - Target FPS: {self.target_fps}')
        self.get_logger().info(f'  - JPEG Quality: {self.jpeg_quality}')
        self.get_logger().info(f'  - Equirect Scale: {self.equirect_scale}')
        self.get_logger().info(f'  - Annotated Scale: {self.annotated_scale}')
        self.get_logger().info(f'  - Publish Equirect: {self.publish_equirect}')

        # Initialize CV bridge
        self.bridge = CvBridge()

        # Set up publishers and subscribers
        self.detection_pub = self.create_publisher(Detection2DArray, 'detections', 10)
        self.annotated_pub = self.create_publisher(CompressedImage, 'annotated_image', 10)
        self.equirect_pub = self.create_publisher(CompressedImage, 'equirectangular_image', 10)
        self.marker_pub = self.create_publisher(MarkerArray, 'detection_markers', 10)
        
        # Subscribe to COMPRESSED image to reduce network traffic
        self.image_sub = self.create_subscription(
            CompressedImage, 
            '/camera/image_raw/compressed', 
            self.image_callback, 
            10
        )

        # Initialize Hailo inference
        self.input_queue = queue.Queue()
        self.output_queue = queue.Queue()
        self.hailo_inference = HailoAsyncInference(
            hef_path=self.model_path,
            input_queue=self.input_queue,
            output_queue=self.output_queue,
        )
        self.model_h, self.model_w, _ = self.hailo_inference.get_input_shape()

        # Initialize tracking and annotation
        self.box_annotator = sv.RoundBoxAnnotator()
        self.label_annotator = sv.LabelAnnotator()
        self.tracker = sv.ByteTrack()

        # Load class names
        with open(self.labels_path, "r", encoding="utf-8") as f:
            self.class_names = f.read().splitlines()
        
        # Create mapping from class name to ID for tracking
        self.class_name_to_id = {name: idx for idx, name in enumerate(self.class_names)}

        # Initialize fisheye camera model if using equirectangular
        self.fisheye_model = None
        if self.use_equirectangular:
            # Will be initialized on first frame when we know image dimensions
            self.get_logger().info('Equirectangular mode enabled')
        
        # Tracking control state
        self.tracking_enabled = False
        self.tracking_object = None
        self.tracking_class_id = None
        
        # Subscribe to tracking control
        self.tracking_control_sub = self.create_subscription(
            String,
            '/tracking_control',
            self.tracking_control_callback,
            10
        )
        
        # Publisher for camera snapshots (for OpenAI)
        self.camera_snapshot_pub = self.create_publisher(
            CompressedImage,
            '/camera_snapshot',
            10
        )
        
        # Store latest annotated frame for snapshot requests
        self.latest_annotated_frame = None

        # Start inference thread
        self.inference_thread = threading.Thread(target=self.hailo_inference.run)
        self.inference_thread.start()
    
    def tracking_control_callback(self, msg):
        """Handle tracking control commands (start:object or stop)"""
        command = msg.data
        
        if command.startswith("start:"):
            obj_name = command.split(":", 1)[1]
            self.tracking_enabled = True
            self.tracking_object = obj_name
            
            # Find class ID for this object
            if obj_name in self.class_name_to_id:
                self.tracking_class_id = self.class_name_to_id[obj_name]
                self.get_logger().info(f'Tracking enabled for: {obj_name} (class_id={self.tracking_class_id})')
            else:
                self.get_logger().warning(f'Unknown object class: {obj_name}. Available classes: {list(self.class_name_to_id.keys())[:10]}...')
                self.tracking_enabled = False
                self.tracking_class_id = None
        
        elif command == "stop":
            self.tracking_enabled = False
            self.tracking_object = None
            self.tracking_class_id = None
            self.get_logger().info('Tracking disabled')

    def image_callback(self, msg):
        # Frame rate limiting - skip frames if processing too fast
        current_time = time.time()
        if current_time - self.last_process_time < self.min_frame_interval:
            return  # Skip this frame
        self.last_process_time = current_time
        
        # Convert COMPRESSED ROS Image to CV2
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        
        # Rotate 180 degrees
        # ONLY UNCOMMENT IF YOUR CAMERA IS UPSIDE DOWN
        # frame = cv2.rotate(frame, cv2.ROTATE_180)

        # Convert to equirectangular FIRST (before color correction)
        # This prevents interpolation artifacts on color-corrected pixels
        if self.use_equirectangular:
            # Initialize fisheye model on first frame
            if self.fisheye_model is None:
                h, w = frame.shape[:2]
                self.fisheye_model = load_camera_model(self.camera_params_path, w, h)
                self.get_logger().info(f'Initialized fisheye model for {w}x{h} image')
            
            # Convert fisheye to equirectangular with optional scaling
            equirect_width = int(frame.shape[1] * self.equirect_scale)
            equirect_height = int(frame.shape[0] * self.equirect_scale)
            
            # Unwarp on original BGR image
            frame = fisheye_to_equirectangular(
                frame, self.fisheye_model, 
                out_width=equirect_width, 
                out_height=equirect_height,
                h_fov_deg=220.0
            )
        
        # Apply color correction AFTER unwarp (on BGR image)
        # Swap r and b channels, then multiply r by 0.5 to fix the colors
        frame = frame[:, :, ::-1]
        frame[:, :, 0] = frame[:, :, 0] * 0.5
        
        # Publish equirectangular image for visualization (only if enabled)
        if self.use_equirectangular and self.publish_equirect:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, jpg_buffer = cv2.imencode('.jpg', frame_bgr, 
                                         [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
            equirect_msg = CompressedImage()
            equirect_msg.format = "jpeg"
            equirect_msg.data = jpg_buffer.tobytes()
            equirect_msg.header = msg.header
            self.equirect_pub.publish(equirect_msg)

        video_h, video_w = frame.shape[:2]

        # Preprocess frame
        preprocessed_frame = self.preprocess_frame(frame, self.model_h, self.model_w, video_h, video_w)

        # Run inference
        self.input_queue.put([preprocessed_frame])
        _, results = self.output_queue.get()

        if len(results) == 1:
            results = results[0]

        # Process detections
        detections = self.extract_detections(results, video_h, video_w, self.score_threshold)

        # Create Detection2DArray message
        detection_msg = Detection2DArray()
        detection_msg.header = msg.header

        # Create MarkerArray message
        marker_array = MarkerArray()

        # Convert detections to ROS messages (filter by tracking if enabled)
        for i in range(detections["num_detections"]):
            class_id = detections["class_id"][i]
            
            # Filter logic:
            # - If tracking is explicitly enabled: filter by tracking_class_id
            # - Otherwise: publish all classes and let the client filter
            if self.tracking_enabled and self.tracking_class_id is not None:
                if class_id != self.tracking_class_id:
                    continue
            
            print("Class ID: ", class_id, f"({self.class_names[class_id]})")
            det = Detection2D()
            det.bbox.center.position.x = float((detections["xyxy"][i][0] + detections["xyxy"][i][2]) / 2)
            det.bbox.center.position.y = float((detections["xyxy"][i][1] + detections["xyxy"][i][3]) / 2)
            det.bbox.size_x = float(detections["xyxy"][i][2] - detections["xyxy"][i][0])
            det.bbox.size_y = float(detections["xyxy"][i][3] - detections["xyxy"][i][1])

            hyp = ObjectHypothesisWithPose()
            hyp.hypothesis.class_id = str(detections["class_id"][i])
            hyp.hypothesis.score = float(detections["confidence"][i])
            det.results.append(hyp)

            detection_msg.detections.append(det)

            # Create marker for bounding box
            marker = Marker()
            marker.header = msg.header
            marker.ns = "detection_boxes"
            marker.id = i
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            marker.scale.x = 0.01  # Line width
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            # Add points to form rectangle
            x1, y1 = float(detections["xyxy"][i][0]), float(detections["xyxy"][i][1])
            x2, y2 = float(detections["xyxy"][i][2]), float(detections["xyxy"][i][3])
            points = [
                (x1, y1, 0.0),
                (x2, y1, 0.0),
                (x2, y2, 0.0),
                (x1, y2, 0.0),
                (x1, y1, 0.0)  # Close the rectangle
            ]
            for x, y, z in points:
                p = Point()
                p.x = x
                p.y = y
                p.z = z
                marker.points.append(p)

            marker_array.markers.append(marker)

        # Publish detections
        self.detection_pub.publish(detection_msg)
        self.marker_pub.publish(marker_array)
        
        # Log detection info for debugging
        if self.tracking_enabled:
            if len(detection_msg.detections) > 0:
                self.get_logger().info(f'🎯 Tracking {self.tracking_object}: {len(detection_msg.detections)} detection(s) | Image size: {video_w}x{video_h}')
            else:
                self.get_logger().debug(f'🔍 Searching for {self.tracking_object} (no detections)')

        # Create and publish annotated image with aggressive optimization
        if detections["num_detections"]:
            annotated_frame = self.postprocess_detections(
                frame, detections, self.class_names, self.tracker,
                self.box_annotator, self.label_annotator
            )
            # Convert back to BGR for encoding
            annotated_bgr = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)
        else:
            # Convert back to BGR for encoding
            annotated_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Scale down annotated image before publishing to reduce bandwidth
        if self.annotated_scale < 1.0:
            h, w = annotated_bgr.shape[:2]
            new_h, new_w = int(h * self.annotated_scale), int(w * self.annotated_scale)
            annotated_bgr = cv2.resize(annotated_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Store latest frame for snapshots (before scaling/compression)
        self.latest_annotated_frame = annotated_bgr.copy()
        
        # Publish snapshot for OpenAI (full quality, no scaling)
        _, snapshot_buffer = cv2.imencode('.jpg', annotated_bgr,
                                          [cv2.IMWRITE_JPEG_QUALITY, 80])  # Higher quality for AI
        snapshot_msg = CompressedImage()
        snapshot_msg.format = "jpeg"
        snapshot_msg.data = snapshot_buffer.tobytes()
        snapshot_msg.header = msg.header
        self.camera_snapshot_pub.publish(snapshot_msg)
        
        # Encode with compression
        _, jpg_buffer = cv2.imencode('.jpg', annotated_bgr, 
                                     [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        annotated_msg = CompressedImage()
        annotated_msg.format = "jpeg"
        annotated_msg.data = jpg_buffer.tobytes()
        annotated_msg.header = msg.header
        self.annotated_pub.publish(annotated_msg)

    def preprocess_frame(
        self, frame: np.ndarray, model_h: int, model_w: int, video_h: int, video_w: int
    ) -> np.ndarray:
        if model_h != video_h or model_w != video_w:
            frame = cv2.resize(frame, (model_w, model_h))
        return frame

    def extract_detections(
        self, hailo_output: List[np.ndarray], h: int, w: int, threshold: float = 0.5
    ) -> Dict[str, np.ndarray]:
        xyxy: List[np.ndarray] = []
        confidence: List[float] = []
        class_id: List[int] = []
        num_detections: int = 0

        for i, detections in enumerate(hailo_output):
            if len(detections) == 0:
                continue
            for detection in detections:
                bbox, score = detection[:4], detection[4]

                if score < threshold:
                    continue

                bbox[0], bbox[1], bbox[2], bbox[3] = (
                    bbox[1] * w,
                    bbox[0] * h,
                    bbox[3] * w,
                    bbox[2] * h,
                )

                xyxy.append(bbox)
                confidence.append(score)
                class_id.append(i)
                num_detections += 1

        return {
            "xyxy": np.array(xyxy),
            "confidence": np.array(confidence),
            "class_id": np.array(class_id),
            "num_detections": num_detections,
        }

    def postprocess_detections(
        self, frame: np.ndarray,
        detections: Dict[str, np.ndarray],
        class_names: List[str],
        tracker: sv.ByteTrack,
        box_annotator: sv.RoundBoxAnnotator,
        label_annotator: sv.LabelAnnotator,
    ) -> np.ndarray:
        sv_detections = sv.Detections(
            xyxy=detections["xyxy"],
            confidence=detections["confidence"],
            class_id=detections["class_id"],
        )

        sv_detections = tracker.update_with_detections(sv_detections)

        labels: List[str] = [
            f"#{tracker_id} {class_names[class_id]}"
            for class_id, tracker_id in zip(sv_detections.class_id, sv_detections.tracker_id)
        ]

        annotated_frame = box_annotator.annotate(
            scene=frame.copy(), detections=sv_detections
        )
        annotated_labeled_frame = label_annotator.annotate(
            scene=annotated_frame, detections=sv_detections, labels=labels
        )
        return annotated_labeled_frame


def main(args=None):
    rclpy.init(args=args)
    node = HailoDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
