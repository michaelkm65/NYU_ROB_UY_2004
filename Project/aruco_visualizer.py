import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
from scipy.spatial.distance import cdist

def get_ideal_path_coordinates(waypoints, cumulative_distances):
    """
    Get X and Y coordinates of the ideal path at given cumulative distances.
    """
    # Build cumulative distance along ideal path
    ideal_distances = [0]
    for i in range(len(waypoints) - 1):
        dist = np.linalg.norm(waypoints[i+1] - waypoints[i])
        ideal_distances.append(ideal_distances[-1] + dist)
    ideal_distances = np.array(ideal_distances)
    
    # Interpolate waypoints based on distance
    ideal_x = np.interp(cumulative_distances, ideal_distances, waypoints[:, 0])
    ideal_y = np.interp(cumulative_distances, ideal_distances, waypoints[:, 1])
    
    return ideal_x, ideal_y

def get_cumulative_distance_along_actual_path(x, y):
    """Calculate cumulative distance traveled along the actual path."""
    distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
    cumulative = np.concatenate([[0], np.cumsum(distances)])
    return cumulative

def point_to_segment_distance(point, seg_start, seg_end):
    """
    Calculate the perpendicular distance from a point to a line segment.
    Returns the distance and the closest point on the segment.
    """
    # Convert to numpy arrays
    p = np.array(point)
    a = np.array(seg_start)
    b = np.array(seg_end)
    
    # Vector from a to b
    ab = b - a
    # Vector from a to p
    ap = p - a
    
    # Project ap onto ab
    ab_squared = np.dot(ab, ab)
    if ab_squared == 0:
        # Segment is a point
        return np.linalg.norm(ap), a
    
    t = np.dot(ap, ab) / ab_squared
    t = np.clip(t, 0, 1)  # Clamp to segment
    
    # Closest point on segment
    closest = a + t * ab
    
    # Distance
    distance = np.linalg.norm(p - closest)
    return distance, closest

def distances_to_ideal_path(positions, waypoints):
    """
    Calculate the minimum distance from each position to the ideal path
    (straight lines connecting waypoints).
    """
    distances = []
    for pos in positions:
        min_dist = float('inf')
        # Check distance to each line segment
        for i in range(len(waypoints) - 1):
            dist, _ = point_to_segment_distance(pos, waypoints[i], waypoints[i+1])
            min_dist = min(min_dist, dist)
        distances.append(min_dist)
    return np.array(distances)

# Load aruco_markers_*.csv (actual path taken) - find most recent file
csv_files = glob.glob("aruco_markers_*.csv")
if csv_files:
    # Get the most recent file
    actual_csv = max(csv_files, key=os.path.getctime)
    print(f"Loaded {actual_csv} with data points")
else:
    # Fallback to Square.csv if it exists
    try:
        actual_csv = 'Square.csv'
        actual_df = pd.read_csv(actual_csv)
        print(f"Loaded {actual_csv} with {len(actual_df)} data points")
    except FileNotFoundError:
        print("No aruco_markers_*.csv or Square.csv files found.")
        exit(1)

try:
    actual_df = pd.read_csv(actual_csv)
    print(f"Loaded {actual_csv} with {len(actual_df)} data points")
except FileNotFoundError:
    print(f"{actual_csv} not found.")
    exit(1)

# Load drawn_points.csv (intended waypoints)
try:
    waypoints_df = pd.read_csv('drawn_points.csv')
    waypoints = waypoints_df[['x_px', 'y_px']].values
    print(f"Loaded drawn_points.csv with {len(waypoints)} waypoints")
except FileNotFoundError:
    waypoints = None
    print("drawn_points.csv not found. Plotting only actual path.")

# Extract actual path data
actual_x = actual_df['center_x'].values
actual_y = actual_df['center_y'].values
angles = actual_df['orientation_angle_deg'].values
timestamps = pd.to_datetime(actual_df['timestamp'])
time_seconds = (timestamps - timestamps.iloc[0]).dt.total_seconds().values

# Find when the robot first reaches a waypoint and trim data
first_waypoint_idx = 0
if waypoints is not None:
    actual_positions = np.column_stack([actual_x, actual_y])
    distances_to_waypoints = cdist(actual_positions, waypoints).min(axis=1)
    
    # Find first time within 30 pixels of a waypoint
    waypoint_threshold = 30
    waypoint_hits = np.where(distances_to_waypoints < waypoint_threshold)[0]
    
    if len(waypoint_hits) > 0:
        first_waypoint_idx = waypoint_hits[0]
        print(f"First waypoint reached at index {first_waypoint_idx} (time: {time_seconds[first_waypoint_idx]:.2f}s)")
    else:
        print("Warning: Robot never reached within 30px of a waypoint")

# Trim data from first waypoint onward
actual_x = actual_x[first_waypoint_idx:]
actual_y = actual_y[first_waypoint_idx:]
angles = angles[first_waypoint_idx:]
time_seconds = time_seconds[first_waypoint_idx:]
time_seconds = time_seconds - time_seconds[0]  # Reset time to start at 0

# Recalculate distance to nearest waypoint for trimmed data
if waypoints is not None:
    actual_positions = np.column_stack([actual_x, actual_y])
    distances_to_waypoints = cdist(actual_positions, waypoints).min(axis=1)
    
    # Calculate distance to ideal path (straight lines connecting waypoints)
    distances_to_ideal_path_values = distances_to_ideal_path(actual_positions, waypoints)
    
    # Calculate cumulative distance along actual path for ideal path interpolation
    cumulative_distance = get_cumulative_distance_along_actual_path(actual_x, actual_y)
    
    # Get ideal path coordinates based on cumulative distance
    ideal_x, ideal_y = get_ideal_path_coordinates(waypoints, cumulative_distance)
    
    # Calculate MSE (Mean Squared Error) - squared distance to nearest waypoint
    mse = np.mean(distances_to_waypoints ** 2)
    rmse = np.sqrt(mse)
    mse_ideal = np.mean(distances_to_ideal_path_values ** 2)
    rmse_ideal = np.sqrt(mse_ideal)
    print(f"Distance to Waypoints - MSE: {mse:.2f}, RMSE: {rmse:.2f}")
    print(f"Distance to Ideal Path - MSE: {mse_ideal:.2f}, RMSE: {rmse_ideal:.2f}")

# Create a comprehensive figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 1. 2D Trajectory Plot (Top Left)
ax1 = plt.subplot(2, 3, 1)
ax1.plot(actual_x, actual_y, 'b-', linewidth=2, alpha=0.7, label='Actual Path')

# Plot ideal path if available
if waypoints is not None:
    ax1.plot(ideal_x, ideal_y, 'orange', linestyle='--', linewidth=2.5, alpha=0.8, label='Ideal Path')

ax1.scatter(actual_x[0], actual_y[0], c='green', s=100, marker='o', label='Start', zorder=5, edgecolors='black', linewidth=2)
ax1.scatter(actual_x[-1], actual_y[-1], c='red', s=100, marker='s', label='End', zorder=5, edgecolors='black', linewidth=2)

ax1.set_xlabel('X Position (pixels)', fontsize=11)
ax1.set_ylabel('Y Position (pixels)', fontsize=11)
ax1.set_title('2D Trajectory: Intended vs Actual Path', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=9, loc='best')
ax1.set_aspect('equal', adjustable='box')

# 2. X Position over Time (Top Middle)
ax2 = plt.subplot(2, 3, 2)
ax2.plot(time_seconds, actual_x, 'b-', linewidth=2, label='Actual X')
if waypoints is not None:
    ax2.plot(time_seconds, ideal_x, 'orange', linestyle='--', linewidth=2, label='Ideal Path X')
ax2.set_xlabel('Time (seconds)', fontsize=11)
ax2.set_ylabel('X Position (pixels)', fontsize=11)
ax2.set_title('X Position Over Time', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=9, loc='best')

# 3. Y Position over Time (Top Right)
ax3 = plt.subplot(2, 3, 3)
ax3.plot(time_seconds, actual_y, 'g-', linewidth=2, label='Actual Y')
if waypoints is not None:
    ax3.plot(time_seconds, ideal_y, 'orange', linestyle='--', linewidth=2, label='Ideal Path Y')
ax3.set_xlabel('Time (seconds)', fontsize=11)
ax3.set_ylabel('Y Position (pixels)', fontsize=11)
ax3.set_title('Y Position Over Time', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=9, loc='best')

# 4. Orientation over Time (Bottom Left)
ax4 = plt.subplot(2, 3, 4)
ax4.plot(time_seconds, angles, 'r-', linewidth=2, label='Orientation')
ax4.set_xlabel('Time (seconds)', fontsize=11)
ax4.set_ylabel('Angle (degrees)', fontsize=11)
ax4.set_title('Marker Orientation Over Time', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=9, loc='best')

# 5. Distance to Ideal Path and MSE (Bottom Middle)
if waypoints is not None:
    ax5 = plt.subplot(2, 3, 5)
    ax5_twin = ax5.twinx()
    
    # Plot distance to ideal path
    line1 = ax5.plot(time_seconds, distances_to_ideal_path_values, 'b-', linewidth=2.5, label='Distance to Ideal Path')
    ax5.fill_between(time_seconds, distances_to_ideal_path_values, alpha=0.2, color='b')
    
    # Plot squared error (for MSE calculation)
    squared_errors_ideal = distances_to_ideal_path_values ** 2
    line2 = ax5_twin.plot(time_seconds, squared_errors_ideal, 'r--', linewidth=2, label='Squared Error (for MSE)')
    
    ax5.set_xlabel('Time (seconds)', fontsize=11)
    ax5.set_ylabel('Distance to Ideal Path (pixels)', fontsize=11, color='b')
    ax5_twin.set_ylabel('Squared Error', fontsize=11, color='r')
    ax5.tick_params(axis='y', labelcolor='b')
    ax5_twin.tick_params(axis='y', labelcolor='r')
    
    ax5.set_title(f'Distance to Ideal Path & MSE\nMSE: {mse_ideal:.2f}, RMSE: {rmse_ideal:.2f}', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Combine legends from ax5 and ax5_twin
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax5.legend(lines, labels, fontsize=9, loc='best')

# 6. Velocity and Statistics (Bottom Right)
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Calculate statistics
displacement = np.sqrt((actual_x[-1] - actual_x[0])**2 + (actual_y[-1] - actual_y[0])**2)
total_distance = np.sum(np.sqrt(np.diff(actual_x)**2 + np.diff(actual_y)**2))
velocity = np.sqrt(np.diff(actual_x)**2 + np.diff(actual_y)**2)
if len(time_seconds) > 1:
    avg_velocity = total_distance / time_seconds[-1]
    max_velocity = velocity.max() / (time_seconds[1] - time_seconds[0]) if len(time_seconds) > 1 else 0
else:
    avg_velocity = 0
    max_velocity = 0

stats_text = f"""
TRAJECTORY STATISTICS
(From first waypoint onwards)

Total Points: {len(actual_x)}
Duration: {time_seconds[-1]:.2f} seconds

POSITION:
  Start: ({actual_x[0]:.1f}, {actual_y[0]:.1f})
  End: ({actual_x[-1]:.1f}, {actual_y[-1]:.1f})
  Displacement: {displacement:.2f} px

DISTANCE:
  Total Distance: {total_distance:.2f} px
  Avg Velocity: {avg_velocity:.2f} px/s
  Max Velocity: {max_velocity:.2f} px/s

ORIENTATION:
  Min Angle: {angles.min():.2f}°
  Max Angle: {angles.max():.2f}°
  Mean Angle: {angles.mean():.2f}°
"""

if waypoints is not None:
    final_distance_ideal = distances_to_ideal_path_values[-1]
    min_distance_ideal = distances_to_ideal_path_values.min()
    avg_distance_ideal = distances_to_ideal_path_values.mean()
    
    stats_text += f"""
TARGET WAYPOINTS:
  Total Waypoints: {len(waypoints)}

DISTANCE TO IDEAL PATH:
  Final Distance: {final_distance_ideal:.2f} px
  Minimum Distance: {min_distance_ideal:.2f} px
  Average Distance: {avg_distance_ideal:.2f} px
  MSE: {mse_ideal:.2f}
  RMSE: {rmse_ideal:.2f}
"""

ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('ArUco Marker Trajectory Analysis - Square Path', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('trajectory_analysis.png', dpi=150, bbox_inches='tight')
print("Saved trajectory analysis to: trajectory_analysis.png")
plt.show()

print("\nVisualization completed!")
