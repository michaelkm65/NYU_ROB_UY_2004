import cv2
import numpy as np
import pandas as pd

# IP address 
url = 'http://10.20.108.207:8080/video'
# url = 'http://10.20.5.63:8080/video'

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
            print(f"\nDetected {len(ids)} ArUco marker(s):")
            
            # Draw detected markers
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            # Print info for each marker
            for i, marker_id in enumerate(ids):
                corner = corners[i][0]
                center_x = int(np.mean(corner[:, 0]))
                center_y = int(np.mean(corner[:, 1]))
                
                # Draw circle at center
                cv2.circle(frame, (center_x, center_y), 8, (0, 255, 255), -1)
                
                # Calculate orientation (angle from center to first corner)
                first_corner = corner[0]
                dx = first_corner[0] - center_x
                dy = first_corner[1] - center_y
                angle = np.arctan2(dy, dx) - np.pi / 4  # Adjust by -45 degrees
                
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
                print(f"  - ArUco ID: {marker_id[0]} | Center: X={center_x}, Y={center_y}")
        else:
            print("No ArUco markers detected")
        
        # Draw points from CSV (always draw, regardless of ArUco detection)
        if drawn_points:
            for i, (x, y) in enumerate(drawn_points):
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
print("\nProgram exited.")
