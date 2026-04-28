import cv2
import numpy as np
from datetime import datetime
from collections import defaultdict
import csv
import subprocess
import sys

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
print("Press 'q' to exit")
print("-" * 60)

while True:
    ret, frame = cap.read()
    if ret:
        # Detect ArUco markers
        corners, ids, rejected = detector.detectMarkers(frame)
        
        # Display detected markers
        if ids is not None:
            print(f"\nFrame {frame_count}: Detected {len(ids)} ArUco marker(s):")
            
            # Draw detected markers
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Print info for each marker
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
                
                # Draw circle at center
                cv2.circle(frame, (center_x, center_y), 8, (0, 255, 255), -1)
                
                # Print info to terminal
                print(f"  - ArUco ID: {mid} | Center: X={center_x}, Y={center_y} | Angle: {angle:.1f}°")
        else:
            print(f"Frame {frame_count}: No ArUco markers detected")
        
        cv2.imshow('ArUco Detection', frame)
        frame_count += 1

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

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
    
    # Run visualizer automatically
    print("\nLaunching visualizer...")
    visualizer_path = "/Users/damianekapanadze/Robotics_NYU_ROB-UY-2004/Project/aruco_visualizer.py"
    subprocess.run([sys.executable, visualizer_path])
else:
    print("\nNo markers detected.")

print("\nProgram exited.")
