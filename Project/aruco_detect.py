import cv2
import numpy as np

# Replace with the IP address displayed on your phone's IP Webcam app
url = 'http://10.20.108.207:8080/video'

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
                
                # Print info to terminal
                print(f"  - ArUco ID: {marker_id[0]} | Center: X={center_x}, Y={center_y}")
        else:
            print("No ArUco markers detected")
        
        cv2.imshow('ArUco Detection', frame)

    # Press 'q' to exit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("\nProgram exited.")
