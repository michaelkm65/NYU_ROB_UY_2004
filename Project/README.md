# Project Template

Skeleton project for writing a closed-loop controller on the Pupper v3
that reacts to objects detected by the onboard Hailo accelerator.

The control loop lives in [`main.py`](main.py). The ROS 2 plumbing
(subscribing to detections, publishing velocity commands, running the camera
and the neural controller) is already wired up.

---

## 1. Running the stack

The robot's base stack (motor controller, camera driver, and object detector)
is launched by a single script:

```bash
./scripts/system_start.sh
```

This script starts, in order:

| Process | What it does |
| --- | --- |
| `ros2 launch robot.launch.py` | Brings up `robot_state_publisher`, the `ros2_control` node with the `neural_controller`, the joint / IMU broadcasters, and the `camera_ros` node publishing raw camera frames. |
| `python hailo_detection.py` | Runs YOLOv5m on the Hailo NPU. Subscribes to the camera and publishes detections on `/detections`, plus annotated and equirectangular images. |

Leave `system_start.sh` running in its own terminal. Then, in a second terminal. 

> **Side note — installing `vision_msgs` if it is missing**
>
> `main.py` imports `vision_msgs/Detection2DArray`. If that package is not
> already available on the robot, you will see an import error on startup.
> Install it into the existing `pupperv3-monorepo/ros2_ws` (which is already
> on `AMENT_PREFIX_PATH` / `PYTHONPATH` in the robot's shell):
>
> **1. Clone the source**
> ```bash
> cd /home/pi/pupperv3-monorepo/ros2_ws/src
> git clone -b jazzy https://github.com/ros-perception/vision_msgs.git
> ```
>
> **2. Install build dependencies**
> ```bash
> sudo rosdep init   # skip if rosdep is already initialized
> rosdep update
> cd /home/pi/pupperv3-monorepo/ros2_ws
> rosdep install --from-paths src/vision_msgs --ignore-src -r -y
> ```
>
> **3. Build**
> ```bash
> cd /home/pi/pupperv3-monorepo/ros2_ws
> source /opt/ros/jazzy/setup.bash
> colcon build --packages-select vision_msgs vision_msgs_py --symlink-install
> ```
---

Your code will need to be added to the main.py:

```bash
python main.py
```

Press **Ctrl+C** in the `main.py` terminal to stop. The script installs a
custom SIGINT handler so that on shutdown it publishes several zero-velocity
commands before tearing down the node — the robot will actually come to a
stop instead of continuing at the last commanded velocity.

### Topics used by `main.py`

| Topic | Direction | Type | Purpose |
| --- | --- | --- | --- |
| `/detections` | subscribed | `vision_msgs/Detection2DArray` | Per-frame object detections from Hailo. |
| `/cmd_vel` | published | `geometry_msgs/Twist` | Body-frame velocity command consumed by the `neural_controller`. |

## 2. Control API (`PupperInterface`)

All ROS 2 interaction is wrapped in a single class, `PupperInterface`, defined
in [`main.py`](main.py). The node is spun on a background thread, so detection
callbacks keep arriving while the main loop sleeps or computes. There is no
need to touch executors, callbacks, or `rclpy.spin()` directly.

The node exposes four things:

### `node.get_detections() -> list[Detection]`

Returns a snapshot of the detections from the **most recent frame**. The list
may be empty if nothing was seen. Each `Detection` is a dataclass with:

| Field | Type | Meaning |
| --- | --- | --- |
| `class_id` | `int` | COCO class id (e.g. `0` = person). |
| `class_name` | `str` | Human-readable label from `coco.txt`, or the numeric id if the label file is missing. |
| `confidence` | `float` | Detection score in `[0, 1]`. |
| `center_x` | `float` | Bounding-box center x in image pixels (image is 700 px wide). |
| `center_y` | `float` | Bounding-box center y in image pixels. |
| `size_x` | `float` | Bounding-box width in pixels. |
| `size_y` | `float` | Bounding-box height in pixels. |
| `normalized_x` | `float` (property) | `center_x` remapped to `[-0.5, +0.5]` — negative is left of image center, positive is right. Useful as a proportional error signal for yaw control. |

Because the camera image is equirectangular, `normalized_x` is roughly linear
in bearing angle, which makes it a good input to a P-controller on
`angular_z`.

### `node.seconds_since_last_detection() -> float`

Returns how long ago (in seconds) the last detection frame arrived. Returns
`float('inf')` if no frame has been received yet. Useful as a watchdog — e.g.
command zero velocity if no frame has arrived for >1 s.

### `node.set_velocity(linear_x=0.0, linear_y=0.0, angular_z=0.0)`

Publishes a single `Twist` on `/cmd_vel`. All arguments are floats in the
robot's body frame:

| Argument | Units | Meaning |
| --- | --- | --- |
| `linear_x` | m/s | Forward (+) / backward (−). |
| `linear_y` | m/s | Strafe left (+) / right (−). |
| `angular_z` | rad/s | Yaw: turn left (+) / right (−). |

The neural controller expects fresh commands. If publishing stops, the
controller holds the last command — this is why the shutdown sequence in
`main()` explicitly publishes zeros before exiting.

### `Detection` dataclass

Constructed by the subscription callback — consumed by the control loop, not
built manually. See [`main.py:36-50`](main.py#L36-L50).

---

## 3. Script structure

[`main.py`](main.py) is organized in three parts:

1. **`Detection` dataclass** ([`main.py:36-50`](main.py#L36-L50))
   Typed container for one detection. Do not modify.

2. **`PupperInterface(Node)`** ([`main.py:53-143`](main.py#L53-L143))
   The ROS 2 node. Owns the publisher, the subscription, the labels file, and
   the thread-safe cache of the latest detections. Treat it as a library — it
   generally should not need modification. To subscribe to an additional
   topic (e.g. IMU, joint states), add the subscription here and expose a
   small getter in the "public API" section of the class.

3. **`main()`** ([`main.py:146-193`](main.py#L146-L193))
   This is where the control logic goes. The template already:
   - initializes `rclpy` (with the default SIGINT handler disabled),
   - constructs the node,
   - spins it on a background thread,
   - installs a Ctrl+C handler that sets a `stop_requested` event,
   - guarantees a clean shutdown that publishes zero velocity several times.

   Fill in the loop body under the `TODO` comment:

   ```python
   while rclpy.ok() and not stop_requested.is_set():
       detections = node.get_detections()
       # ... pick a target, compute a control command ...
       node.set_velocity(linear_x=..., linear_y=..., angular_z=...)
       time.sleep(0.1)
   ```

   Keep the `time.sleep(0.1)` (or similar) in the loop — it yields the GIL
   to the spin thread and sets the control rate. A 10 Hz control loop is a
   reasonable default; the detector runs at ~5 Hz, so going much faster just
   repeats the same detections.

   **Do not** call `rclpy.spin()` manually, and **do not** remove the
   `stop_requested` check — it is what lets Ctrl+C exit the loop cleanly so
   the finally-block can stop the robot.
