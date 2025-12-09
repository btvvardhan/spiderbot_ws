# 🕷️ Spiderbot - Bio-Inspired Quadruped Robot

<div align="center">

**A 12-DOF bio-inspired spider robot with CPG-RL locomotion, SLAM capabilities, and VLM-powered navigation**

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

[Features](#-features) • [Hardware](#-hardware-requirements) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Usage](#-usage-modes) • [Troubleshooting](#-troubleshooting)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Hardware Requirements](#-hardware-requirements)
- [Software Prerequisites](#-software-prerequisites)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage Modes](#-usage-modes)
- [Project Structure](#-project-structure)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)

---

## 🌟 Overview

This project implements a bio-inspired quadruped spider robot featuring:

- **Hybrid CPG-RL Control**: Combines Central Pattern Generators (Hopf oscillators) with Deep Reinforcement Learning
- **Omnidirectional Movement**: 6-DOF control (forward, backward, lateral, rotation) from a forward-trained policy using dynamic sign control
- **Sensor-less Control**: IMU-only feedback (no joint encoders required)
- **SLAM Integration**: 2D mapping and localization using LiDAR
- **VLM Navigation**: Natural language control via Google Gemini 2.0 Flash
- **Sim-to-Real Transfer**: Trained in Isaac Lab, deployed on real hardware

---

## ✨ Features

| Category | Features |
|----------|----------|
| **Locomotion** | • Hopf CPG with diagonal coupling<br>• PPO-trained policy (95%+ success rate)<br>• Omnidirectional movement via dynamic coxa sign control<br>• Multiple gaits (walk, trot, pace) |
| **Hardware** | • 12 DOF (3 joints per leg: coxa, femur, tibia)<br>• DS3225 servos with PCA9685 drivers<br>• Jetson Nano for policy inference<br>• Arduino Mega for servo control |
| **Sensing** | • MPU9250 IMU (9-axis)<br>• YDLidar X2 (2D LIDAR)<br>• Logitech Brio camera<br>• Optional: Raspberry Pi for sensor hub |
| **Navigation** | • SLAM Toolbox for mapping<br>• EKF sensor fusion<br>• Vision-Language Model dashboard<br>• Natural language command interface |
| **Simulation** | • Gazebo Ignition support<br>• ROS2 Control integration<br>• RViz visualization<br>• Joint trajectory control |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HIGH-LEVEL CONTROL                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │  Teleop    │  │ VLM Gemini │  │    SLAM    │            │
│  │ (Keyboard) │  │ Dashboard  │  │  Mapping   │            │
│  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘            │
│         │                │                │                  │
│         └────────────────┴────────────────┘                  │
│                          │                                   │
│                    /cmd_vel                                  │
└──────────────────────────┼───────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────┐
│                  ROS2 CONTROL LAYER                          │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  Policy Node (policy_omni_node / policy_cpg_node)    │   │
│  │  • Loads trained PyTorch policy                      │   │
│  │  • Observes: cmd_vel, IMU, CPG phase                 │   │
│  │  • Outputs: CPG parameters (freq, amp, phase)        │   │
│  └────────────────────┬──────────────────────────────────┘   │
│                       │                                       │
│  ┌────────────────────▼──────────────────────────────────┐   │
│  │  Hopf CPG Layer                                       │   │
│  │  • 4 coupled oscillators (FL, FR, RL, RR)            │   │
│  │  • Diagonal phase coupling (k=0.7)                   │   │
│  │  • Generates joint position deltas                   │   │
│  └────────────────────┬──────────────────────────────────┘   │
│                       │                                       │
│                 /joint_states                                 │
└───────────────────────┼───────────────────────────────────────┘
                        │
┌───────────────────────▼───────────────────────────────────────┐
│                 HARDWARE INTERFACE                            │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  Serial Bridge (serial_bridge_jointstate.py)         │    │
│  │  • Converts joint_states → servo angles              │    │
│  │  • Remaps joint order for hardware                   │    │
│  │  • Publishes to /dev/ttyACM0 @ 115200 baud          │    │
│  └────────────────────┬──────────────────────────────────┘    │
└────────────────────────┼──────────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────────┐
│                 ARDUINO MEGA                                  │
│  ┌──────────────────────────────────────────────────────┐    │
│  │  spiderbot_firmware.ino                               │    │
│  │  • Receives CSV angles (12 values)                   │    │
│  │  • Controls 2× PCA9685 (12 servos)                   │    │
│  │  • Reads MPU9250 IMU @ 20Hz                          │    │
│  │  • Non-blocking state machine                        │    │
│  └────────────────────┬──────────────────────────────────┘    │
└────────────────────────┼──────────────────────────────────────┘
                         │
┌────────────────────────▼──────────────────────────────────────┐
│                    HARDWARE                                   │
│  • 12× DS3225 Servos                                         │
│  • 2× PCA9685 PWM Drivers (I2C: 0x40, 0x41)                 │
│  • MPU9250 IMU (I2C: 0x68)                                   │
│  • YDLidar X2 (USB)                                          │
│  • Logitech Brio Camera (USB)                                │
└───────────────────────────────────────────────────────────────┘
```

---

## 🔧 Hardware Requirements

### **Main Computer** (for high-level control)
- **CPU**: Intel Core i5 or better
- **RAM**: 16 GB minimum
- **GPU**: Optional (NVIDIA recommended for training)
- **OS**: Ubuntu 22.04 LTS

### **Onboard Computer** (Jetson Nano or equivalent)
- **Model**: Jetson Nano 4GB or Xavier NX
- **Storage**: 64GB SD card minimum
- **Accessories**: Cooling fan required

### **Microcontroller**
- **Arduino Mega 2560** (for servo control)
- **USB Cable**: Type B for Arduino

### **Actuators**
- **Servos**: 12× DS3225 (25kg-cm, metal gear)
- **Drivers**: 2× PCA9685 16-channel PWM driver boards
- **Power**: 5V 10A+ power supply

### **Sensors**
- **IMU**: MPU9250 (9-DOF, I2C)
- **LiDAR**: YDLidar X2 (2D, 8m range, USB)
- **Camera**: Logitech Brio or similar (USB)

### **Optional**
- **Raspberry Pi 4** (sensor hub for distributed architecture)
- **WiFi Router** (for ROS2 network)

---

## 💻 Software Prerequisites

### **Operating System**
```bash
Ubuntu 22.04 LTS (Jammy Jellyfish)
```

### **ROS2**
```bash
ROS 2 Humble Hawksbill
```

### **Python Packages**
```bash
python3.10 or python3.11
pytorch
numpy
opencv-python
pyserial
google-generativeai  # For VLM
flask  # For VLM dashboard
```

### **System Dependencies**
```bash
sudo apt install -y \
    ros-humble-desktop-full \
    ros-humble-gazebo-ros-pkgs \
    ros-humble-ros2-control \
    ros-humble-ros-gz \
    ros-humble-slam-toolbox \
    ros-humble-robot-localization \
    python3-pip \
    python3-colcon-common-extensions
```

---

## 🚀 Installation

### **Step 1: Clone Repository**

```bash
cd ~
git clone <your-repo-url> spiderbot_ws
cd spiderbot_ws
```

### **Step 2: Install ROS2 Dependencies**

```bash
cd ~/spiderbot_ws
rosdep install --from-paths src --ignore-src -r -y
```

### **Step 3: Install Python Dependencies**

```bash
pip3 install torch numpy opencv-python pyserial google-generativeai flask flask-cors pillow
```

### **Step 4: Build Workspace**

```bash
cd ~/spiderbot_ws
colcon build --symlink-install
source install/setup.bash
```

### **Step 5: Arduino Setup**

1. **Install Arduino IDE**:
```bash
wget https://downloads.arduino.cc/arduino-1.8.19-linux64.tar.xz
tar -xf arduino-1.8.19-linux64.tar.xz
sudo mv arduino-1.8.19 /opt/
cd /opt/arduino-1.8.19
sudo ./install.sh
```

2. **Install Libraries** (via Arduino IDE Library Manager):
   - Adafruit PWM Servo Driver Library
   - Wire (built-in)

3. **Upload Firmware**:
   - Open: `src/spiderbot_arduino/spiderbot_firmware/spiderbot_firmware.ino`
   - Board: Arduino Mega 2560
   - Port: `/dev/ttyACM0`
   - Upload

### **Step 6: Set Permissions**

```bash
# Arduino serial port
sudo usermod -a -G dialout $USER
sudo chmod 666 /dev/ttyACM0

# Logout and login for group changes to take effect
```

### **Step 7: Verify Installation**

```bash
# Check ROS2
ros2 pkg list | grep spiderbot

# Should see:
# spiderbot_control
# spiderbot_description
# spiderbot_slam
# spiderbot_vlm
```

---

## ⚡ Quick Start

### **Test 1: Verify Arduino Communication**

```bash
# Terminal 1: Monitor Arduino
python3 -m serial.tools.miniterm /dev/ttyACM0 115200

# Should see:
# BOOT
# READY
# IMU_OK (if IMU connected)
```

### **Test 2: Home Position (All servos to 90°)**

```bash
# Upload and run: src/spiderbot_arduino/90home/90home.ino
# All servos should move to neutral position
```

### **Test 3: Visualization in RViz**

```bash
source ~/spiderbot_ws/install/setup.bash
ros2 launch spiderbot_description view_robot.launch.py
```

---

## 🎮 Usage Modes

## **Mode 1: Simulation (Gazebo)**

Launch complete simulation environment:

```bash
source ~/spiderbot_ws/install/setup.bash
ros2 launch spiderbot_description sim_gz.launch.py
```

**What it does:**
- Spawns robot in Gazebo Ignition
- Loads joint controllers
- Starts RViz visualization

**Control the robot:**
```bash
# Terminal 2: Policy control
ros2 run spiderbot_control policy_omni_node

# Terminal 3: Keyboard teleop
ros2 run spiderbot_control teleop_keyboard
```

**Controls:**
- `W/S`: Forward/Backward
- `A/D`: Strafe Left/Right
- `Q/E`: Rotate Left/Right
- `SPACE`: Emergency stop

---

## **Mode 2: Hardware Control (Trained Policy)**

### **Option A: Forward Motion Only**

```bash
# Terminal 1: Serial bridge (ROS → Arduino)
source ~/spiderbot_ws/install/setup.bash
ros2 run spiderbot_control serial_bridge_jointstate

# Terminal 2: Policy controller
ros2 run spiderbot_control policy_cpg_node \
    --ros-args \
    -p policy_pt:=/path/to/your/policy.pt

# Terminal 3: Keyboard teleop
ros2 run spiderbot_control teleop_keyboard
```

### **Option B: Omnidirectional Control**

Uses dynamic coxa sign control for full 6-DOF movement:

```bash
# Terminal 1: Serial bridge
ros2 run spiderbot_control serial_bridge_jointstate

# Terminal 2: Omnidirectional policy
ros2 run spiderbot_control policy_omni_node \
    --ros-args \
    -p policy_pt:=/path/to/your/policy.pt

# Terminal 3: Keyboard teleop
ros2 run spiderbot_control teleop_keyboard
```

---

## **Mode 3: Hardcoded Gaits (No Policy)**

Test pre-programmed gait patterns without RL policy:

### **Option A: Arduino-Only Control**

```bash
# Upload: src/spiderbot_arduino/spiderbot_hardcode/spiderbot_hardcode.ino
# Open Serial Monitor (115200 baud)
# Send commands:
# w = forward, s = backward
# a = left, d = right
# q = yaw left, e = yaw right
# x = stop, +/- = speed
```

### **Option B: ROS2 Hardcoded Gaits**

```bash
# Terminal 1: Serial bridge
ros2 run spiderbot_control serial_bridge_jointstate

# Terminal 2: Hardcoded gait node
ros2 run spiderbot_control fk

# Terminal 3: Teleop
ros2 run spiderbot_control teleop_keyboard
```

**This mode is useful for:**
- Testing hardware without trained policy
- Debugging servo connectivity
- Verifying mechanical assembly

---

## **Mode 4: SLAM Mapping**

Create maps of your environment using LiDAR and IMU:

### **Gazebo Simulation**

```bash
source ~/spiderbot_ws/install/setup.bash
ros2 launch spiderbot_slam slam_gazebo.launch.py
```

### **Physical Robot**

```bash
# Terminal 1: SLAM + Odometry + Visualization
ros2 launch spiderbot_slam slam_minimal.launch.py

# Terminal 2: Policy control
ros2 run spiderbot_control policy_omni_node

# Terminal 3: Teleop
ros2 run spiderbot_control teleop_keyboard
```

**SLAM Topics:**
- `/scan` - LiDAR data (LaserScan)
- `/imu` - IMU data (Imu)
- `/map` - Generated map (OccupancyGrid)
- `/odom` - Odometry estimate (Odometry)

**Save Map:**
```bash
ros2 run nav2_map_server map_saver_cli -f ~/my_map
```

---

## **Mode 5: VLM Navigation (Vision-Language Model)**

Control robot using natural language via Google Gemini:

### **Prerequisites**

1. **Get Gemini API Key**: https://aistudio.google.com/apikey

2. **Set Environment Variable**:
```bash
export GEMINI_API_KEY="your_key_here"
# Add to ~/.bashrc to persist:
echo 'export GEMINI_API_KEY="your_key_here"' >> ~/.bashrc
```

### **Launch VLM System**

```bash
source ~/spiderbot_ws/install/setup.bash
ros2 launch spiderbot_vlm vlm_full.launch.py
```

**Access Dashboard:** http://localhost:5000

**What it does:**
1. Displays live camera feed
2. Accepts natural language commands
3. Uses Gemini Vision to:
   - Describe scenes
   - Plan movement sequences
   - Generate WASD-style commands
4. Executes commands on robot

**Example Commands:**
- "Describe what you see"
- "Go to the table"
- "Turn left and move forward"
- "Avoid the obstacle on the right"
- "Stop"

---

## **Mode 6: Arduino Serial Control Only**

Direct control via Arduino firmware for testing:

### **Option A: Basic Firmware (IMU + Servos)**

```bash
# Upload: src/spiderbot_arduino/spiderbot_firmware/spiderbot_firmware.ino
# This receives joint angles via serial and publishes IMU data
```

**Send angles via Python:**
```python
import serial
import time

ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)
time.sleep(2)

# Send 12 joint angles (0-180°) as CSV
angles = "90,90,90,90,90,90,90,90,90,90,90,90\n"
ser.write(angles.encode())
ser.close()
```

### **Option B: Diagnostic Mode**

```bash
# Upload: src/spiderbot_arduino/diagnostic_code/diagnostic_code.ino
# Echoes all received data for debugging
```

---

## **Mode 7: Control via ASCII Commands**

Simple character-based control:

```bash
# Terminal 1: Hardcoded gaits with ASCII
# Upload: src/spiderbot_arduino/spiderbot_hardcode/spiderbot_hardcode.ino

# Terminal 2: ROS2 control wrapper
ros2 run spiderbot_control control

# Sends: 'w', 's', 'a', 'd', 'q', 'e', 'x' to /cmd topic
```

---

## 📁 Project Structure

```
spiderbot_ws/
├── src/
│   ├── spiderbot_arduino/          # Arduino firmware
│   │   ├── 90home/                 # Home position test
│   │   ├── diagnostic_code/        # Serial debugging
│   │   ├── spiderbot_firmware/     # Main firmware (IMU + Servos)
│   │   ├── spiderbot_hardcode/     # Hardcoded gaits
│   │   └── spiderbot_serial_bridge/# Simple serial control
│   │
│   ├── spiderbot_control/          # ROS2 control package
│   │   ├── launch/
│   │   │   └── spiderbot_cpgrl.launch.py
│   │   ├── spiderbot_control/
│   │   │   ├── cpg.py              # Hopf CPG implementation
│   │   │   ├── policy_cpg_node.py  # Forward policy node
│   │   │   ├── policy_omni_node.py # Omnidirectional policy node
│   │   │   ├── fk.py               # Hardcoded gait node
│   │   │   ├── teleop_keyboard.py  # Keyboard teleop
│   │   │   ├── control.py          # ASCII command publisher
│   │   │   └── serial_bridge_jointstate.py # ROS→Arduino bridge
│   │   └── models/
│   │       └── policy.pt           # Trained policy (add your own)
│   │
│   ├── spiderbot_description/      # Robot description
│   │   ├── urdf/
│   │   │   ├── spidy.urdf          # Robot URDF
│   │   │   └── spidy.xacro         # Xacro template
│   │   ├── meshes/                 # STL files
│   │   ├── launch/
│   │   │   ├── view_robot.launch.py    # RViz visualization
│   │   │   └── sim_gz.launch.py        # Gazebo simulation
│   │   └── config/
│   │       └── controllers.yaml    # ROS2 Control config
│   │
│   ├── spiderbot_slam/             # SLAM package
│   │   ├── config/
│   │   │   ├── slam_toolbox.yaml   # SLAM parameters
│   │   │   ├── ekf_config.yaml     # EKF fusion config
│   │   │   └── slam_view.rviz      # RViz config
│   │   ├── launch/
│   │   │   ├── slam_gazebo.launch.py   # Simulation SLAM
│   │   │   ├── slam_minimal.launch.py  # Physical robot SLAM
│   │   │   └── ekf.launch.py           # EKF standalone
│   │   ├── scripts/
│   │   │   └── minimal_imu_odom.py # IMU odometry node
│   │   └── worlds/
│   │       └── test_world.sdf      # Gazebo world
│   │
│   └── spiderbot_vlm/              # Vision-Language Model
│       ├── spiderbot_vlm/
│       │   ├── vlm_scene_planner_node.py  # Gemini planner
│       │   └── vlm_dashboard_node.py      # Web dashboard
│       ├── templates/
│       │   └── Dashboard.html      # Web UI
│       └── launch/
│           └── vlm_full.launch.py  # Complete VLM system
```

---

## 🔧 Troubleshooting

### **Arduino Issues**

**Problem**: Arduino not detected
```bash
# Check connection
ls /dev/ttyACM*

# If not found:
sudo apt install arduino-core
sudo usermod -a -G dialout $USER
# Logout and login
```

**Problem**: Upload failed
```bash
# Check board selection: Tools → Board → Arduino Mega 2560
# Check port: Tools → Port → /dev/ttyACM0
# Try different USB cable
```

**Problem**: Servos not moving
```bash
# Check serial monitor (115200 baud):
# Should see: BOOT, READY
# Check power supply (5V 10A minimum)
# Check PCA9685 I2C addresses (0x40, 0x41)
```

### **ROS2 Issues**

**Problem**: Package not found
```bash
# Rebuild workspace
cd ~/spiderbot_ws
colcon build --symlink-install
source install/setup.bash

# Verify
ros2 pkg list | grep spiderbot
```

**Problem**: Serial port permission denied
```bash
sudo chmod 666 /dev/ttyACM0
sudo usermod -a -G dialout $USER
# Logout and login
```

**Problem**: joint_states not publishing
```bash
# Check serial bridge is running:
ros2 node list | grep serial_bridge

# Check topics:
ros2 topic list | grep joint_states
ros2 topic echo /joint_states
```

### **Gazebo Issues**

**Problem**: Meshes not loading
```bash
# Check IGN_GAZEBO_RESOURCE_PATH
echo $IGN_GAZEBO_RESOURCE_PATH

# Should include workspace path
# If not, add to ~/.bashrc:
export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:~/spiderbot_ws/install
```

**Problem**: Controllers not loading
```bash
# Check controller spawner:
ros2 control list_controllers

# Reload if needed:
ros2 launch spiderbot_description sim_gz.launch.py
```

### **SLAM Issues**

**Problem**: No map building
```bash
# Check topics:
ros2 topic hz /scan  # Should be ~10Hz
ros2 topic hz /odom  # Should be ~50Hz

# Check TF tree:
ros2 run tf2_tools view_frames
# Should see: map → odom → base_link → laser_frame
```

**Problem**: Robot drifting in map
```bash
# Increase loop closure:
# Edit: src/spiderbot_slam/config/slam_toolbox.yaml
# Set: loop_search_maximum_distance: 5.0
```

### **VLM Issues**

**Problem**: Dashboard not accessible
```bash
# Check if Flask is running:
ps aux | grep vlm_dashboard_node

# Check port:
netstat -tulpn | grep 5000

# Try different port:
ros2 run spiderbot_vlm vlm_dashboard_node --ros-args -p port:=5001
```

**Problem**: Gemini API error
```bash
# Check API key:
echo $GEMINI_API_KEY

# Verify key at: https://aistudio.google.com/apikey

# Check quota:
# https://console.cloud.google.com/apis/api/generativelanguage.googleapis.com/quotas
```

### **Policy Issues**

**Problem**: Robot falls immediately
```bash
# Check policy path:
ls /path/to/policy.pt

# Use hardcoded gaits instead:
ros2 run spiderbot_control fk
```

**Problem**: Erratic movement
```bash
# Check joint signs in policy_omni_node.py:
# joint_signs parameter should match your robot
# Try policy_cpg_node instead for forward-only
```


---

## 📄 License

Apache License 2.0

---

<div align="center">

**🕷️ Built with passion for bio-inspired robotics 🕷️**

[⬆ Back to Top](#-spiderbot---bio-inspired-quadruped-robot)

</div>
