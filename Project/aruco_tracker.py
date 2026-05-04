import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import csv

# Replace with the IP address displayed on your phone's IP Webcam app
# url = 'http://10.20.108.207:8080/video'
url = 'http://10.20.5.63:8080/video'

# Load drawn points from CSV
try:
    points_df = pd.read_csv('drawn_points.csv')
    drawn_points = list(zip(points_df['x_px'].astype(int), points_df['y_px'].astype(int)))
    print(f"Loaded {len(drawn_points)} points from drawn_points.csv")
except FileNotFoundError:
    drawn_points = []
    print("drawn_points.csv not found")
except Exception as e:
    drawn_points = []
    print(f"Error loading CSV: {e}")

# Skip every Nth point (set to 1 for no skipping, 2 for every other, 3 for every third, etc.)
WAYPOINT_SKIP = 5  # Change this to skip more or fewer points
filtered_drawn_points = drawn_points[::WAYPOINT_SKIP]
if WAYPOINT_SKIP > 1:
    print(f"Skipping every {WAYPOINT_SKIP} points. Active waypoints: {len(filtered_drawn_points)} out of {len(drawn_points)}")

# Initialize waypoint tracking
target_index = 0
waypoint_threshold = 30  # Distance threshold to skip to next waypoint (pixels)

# Initialize CSV file for ArUco marker recording (will be created when first marker is detected)
aruco_csv_filename = f"aruco_markers_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
aruco_csv_file = None
aruco_csv_writer = None
csv_initialized = False

# Initialize ArUco detector
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

cap = cv2.VideoCapture(url)

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
            # Initialize CSV file on first detection
            if not csv_initialized:
                aruco_csv_file = open(aruco_csv_filename, 'w', newline='')
                aruco_csv_writer = csv.writer(aruco_csv_file)
                aruco_csv_writer.writerow(['timestamp', 'marker_id', 'center_x', 'center_y', 'orientation_angle_rad', 'orientation_angle_deg'])
                csv_initialized = True
                print(f"ArUco markers will be recorded to: {aruco_csv_filename}")
            
            print(f"\nDetected {len(ids)} ArUco marker(s):")
            
            # Draw detected markers
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Print info for each marker
            for i, marker_id in enumerate(ids):
                corner = corners[i][0]
                center_x = int(np.mean(corner[:, 0]))
                center_y = int(np.mean(corner[:, 1]))
                
                # Calculate orientation (angle from center to first corner)
                first_corner = corner[0]
                dx = first_corner[0] - center_x
                dy = first_corner[1] - center_y
                angle = np.arctan2(dy, dx) - np.pi / 4  # Adjust by -45 degrees
                
                # Check distance to current target waypoint and advance if close
                if filtered_drawn_points and target_index < len(filtered_drawn_points):
                    target_x, target_y = filtered_drawn_points[target_index]
                    dist = np.sqrt((center_x - target_x)**2 + (center_y - target_y)**2)
                    
                    # Skip waypoints when marker gets close
                    if dist < waypoint_threshold and target_index < len(filtered_drawn_points) - 1:
                        target_index += 1
                        print(f"  → Waypoint reached! Moving to waypoint {target_index + 1}/{len(filtered_drawn_points)}")
                
                # Record to CSV (only records when markers are detected)
                timestamp = datetime.now().isoformat()
                angle_deg = np.degrees(angle)
                aruco_csv_writer.writerow([timestamp, marker_id[0], center_x, center_y, angle, angle_deg])
                aruco_csv_file.flush()  # Flush to ensure data is written
                
                # Draw circle at center
                cv2.circle(frame, (center_x, center_y), 8, (0, 255, 255), -1)
                
                # Draw coordinate axes (X and Y) at marker center
                axis_length = 50
                
                # X-axis (red) - rotated by marker angle
                x_end_x = int(center_x + axis_length * np.cos(angle))
                x_end_y = int(center_y + axis_length * np.sin(angle))
                cv2.arrowedLine(frame, (center_x, center_y), (x_end_x, x_end_y), (0, 0, 255), 2, tipLength=0.3)
                
                # Y-axis (green) - perpendicular to X-axis
                y_angle = angle + np.pi / 2
                y_end_x = int(center_x + axis_length * np.cos(y_angle))
                y_end_y = int(center_y + axis_length * np.sin(y_angle))
                cv2.arrowedLine(frame, (center_x, center_y), (y_end_x, y_end_y), (0, 255, 0), 2, tipLength=0.3)
                
                # Print info to terminal
                print(f"  - ArUco ID: {marker_id[0]} | Center: X={center_x}, Y={center_y} | Angle: {angle_deg:.2f}°")
        else:
            print("No ArUco markers detected")
        
        # Draw points from CSV (always draw, regardless of ArUco detection)
        if filtered_drawn_points:
            for i, (x, y) in enumerate(filtered_drawn_points):
                # Draw circle for each point
                cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Blue filled circle
                # Draw a small outline
                cv2.circle(frame, (x, y), 7, (255, 255, 0), 2)  # Cyan outline
        
        # Add coordinate labels at frame corners
        height, width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_color = (255, 255, 255)  # White
        thickness = 1

        #flip the stream output horizontally to match the coordinate system of the drawn points
        frame = cv2.flip(frame, 0)        
        # Top-left corner (0, 0)
        cv2.putText(frame, "(0, 0)", (5, 20), font, font_scale, font_color, thickness)
        
        # Top-right corner (width, 0)
        text_size = cv2.getTextSize(f"({width}, 0)", font, font_scale, thickness)[0]
        cv2.putText(frame, f"({width}, 0)", (width - text_size[0] - 5, 20), font, font_scale, font_color, thickness)
        
        # Bottom-left corner (0, height)
        cv2.putText(frame, f"(0, {height})", (5, height - 5), font, font_scale, font_color, thickness)
        
        # Bottom-right corner (width, height)
        text_size = cv2.getTextSize(f"({width}, {height})", font, font_scale, thickness)[0]
        cv2.putText(frame, f"({width}, {height})", (width - text_size[0] - 5, height - 5), font, font_scale, font_color, thickness)
        
        cv2.imshow('ArUco Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
if csv_initialized:
    aruco_csv_file.close()
    print(f"\nProgram exited. ArUco data saved to: {aruco_csv_filename}")
else:
    print(f"\nProgram exited. No ArUco markers were detected.")
