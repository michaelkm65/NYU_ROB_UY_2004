import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
import csv

# Replace with the IP address displayed on your phone's IP Webcam app
url = 'http://10.20.110.172:8080/video'

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

cap = cv2.VideoCapture(url)

# Data storage for positions
marker_positions = defaultdict(list)  # marker_id -> [[x, y, frame], ...]
frame_count = 0

print("Starting ArUco Detection...")
print("Press Ctrl+C to exit")
print("-" * 60)

try:
    while True:
        ret, frame = cap.read()
        if ret:
            # Detect ArUco markers
            corners, ids, rejected = detector.detectMarkers(frame)
            
            # Process detected markers
            if ids is not None:
                print(f"\nFrame {frame_count}: Detected {len(ids)} marker(s):")
                
                # Process each marker
                for i, marker_id in enumerate(ids):
                    corner = corners[i][0]
                    center_x = int(np.mean(corner[:, 0]))
                    center_y = int(np.mean(corner[:, 1]))
                    
                    # Calculate angle from corner positions
                    dx = corner[1][0] - corner[0][0]
                    dy = corner[1][1] - corner[0][1]
                    angle = np.arctan2(dy, dx) * 180 / np.pi
                    
                    # Store position data with angle
                    mid = marker_id[0]
                    marker_positions[mid].append([center_x, center_y, angle, frame_count])
                    
                    # Print info to terminal
                    print(f"  - ID: {mid} | X={center_x}, Y={center_y} | Angle: {angle:.1f}°")
            else:
                # Print status every 100 frames
                if frame_count % 100 == 0:
                    print(f"Frame {frame_count}: No markers detected")
            
            frame_count += 1
        else:
            print("Failed to read frame from stream")
            break

except KeyboardInterrupt:
    print("\n\nExiting...")

finally:
    cap.release()
    
    # Save positions to CSV file
    if marker_positions:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aruco_positions_{timestamp}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Write header
            writer.writerow(['Marker_ID', 'Frame', 'X', 'Y', 'Angle_degrees'])
            
            # Write data for each marker
            for marker_id in sorted(marker_positions.keys()):
                positions = marker_positions[marker_id]
                for x, y, angle, frame in positions:
                    writer.writerow([marker_id, frame, x, y, f"{angle:.2f}"])
        
        print(f"\n✓ Positions saved to: {filename}")
        print(f"Total frames: {frame_count}")
        print(f"Markers detected: {sorted(marker_positions.keys())}")
    else:
        print("\nNo markers detected.")
    
    print("\nProgram exited.")
