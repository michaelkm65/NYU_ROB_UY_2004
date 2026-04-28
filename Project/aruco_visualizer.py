import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyArrowPatch
import numpy as np
import glob
import os

# Find the most recent CSV file
csv_files = glob.glob("aruco_positions_*.csv")
if not csv_files:
    print("No CSV files found. Run aruco_tracker.py first.")
    exit(1)

# Get the most recent file
csv_file = max(csv_files, key=os.path.getctime)
print(f"Loading data from: {csv_file}")

# Read CSV file
df = pd.read_csv(csv_file)

# Group by marker ID
for marker_id in df['Marker_ID'].unique():
    marker_df = df[df['Marker_ID'] == marker_id].reset_index(drop=True)
    
    print(f"\nProcessing Marker ID: {marker_id}")
    print(f"Total points: {len(marker_df)}")
    
    # Extract data
    x = marker_df['X'].values
    y = marker_df['Y'].values
    angles = marker_df['Angle_degrees'].values
    frames = marker_df['Frame'].values
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot the trajectory path
    ax.plot(x, y, 'b-', linewidth=2, label='Trajectory', alpha=0.6)
    ax.scatter(x[0], y[0], c='green', s=150, marker='o', label='Start', zorder=5)
    ax.scatter(x[-1], y[-1], c='red', s=150, marker='s', label='End', zorder=5)
    
    # Set axis properties
    ax.set_xlabel('X Position (pixels)', fontsize=12)
    ax.set_ylabel('Y Position (pixels)', fontsize=12)
    ax.set_title(f'ArUco Marker ID {marker_id} - Position Tracking', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    
    # Initialize arrow
    arrow = FancyArrowPatch((x[0], y[0]), (x[0], y[0]), 
                           arrowstyle='->', mutation_scale=30, 
                           linewidth=3, color='red', zorder=10)
    ax.add_patch(arrow)
    
    # Text annotations
    text_pos = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                      verticalalignment='top', fontsize=10,
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Animation update function
    def animate(frame_idx):
        if frame_idx < len(x):
            curr_x = x[frame_idx]
            curr_y = y[frame_idx]
            angle = angles[frame_idx]
            
            # Calculate arrow endpoint based on angle
            arrow_length = 50
            end_x = curr_x + arrow_length * np.cos(angle * np.pi / 180)
            end_y = curr_y + arrow_length * np.sin(angle * np.pi / 180)
            
            # Update arrow
            arrow.set_positions((curr_x, curr_y), (end_x, end_y))
            
            # Update text
            text_pos.set_text(f'Frame: {int(frames[frame_idx])}\n'
                            f'Position: ({curr_x:.1f}, {curr_y:.1f})\n'
                            f'Angle: {angle:.1f}°')
        
        return arrow, text_pos
    
    # Create animation
    anim = animation.FuncAnimation(fig, animate, frames=len(x), 
                                 interval=50, blit=True, repeat=True)
    
    # Save animation
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    animation_file = f"aruco_marker{marker_id}_animation_{timestamp}.gif"
    print(f"Saving animation to: {animation_file}")
    anim.save(animation_file, writer='pillow', fps=20)
    
    plt.tight_layout()
    plt.show()

print("\nAnimation completed!")
