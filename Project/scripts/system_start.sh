#!/bin/bash
# Start the robot policy, camera node, and object detector.

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
# Project root is one level up from scripts directory
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Change to project directory
cd "$PROJECT_ROOT"

source ~/.bashrc
ros2 launch "$PROJECT_ROOT/robot.launch.py" &
python "$PROJECT_ROOT/hailo_detection.py"
