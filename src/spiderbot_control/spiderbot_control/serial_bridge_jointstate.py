#!/usr/bin/env python3
"""
ROS2 node: subscribe to /joint_states and send CSV over serial to Arduino Mega.

Features:
 - Configurable serial port & baud
 - Per-joint mapping from joint name -> channel index (0..N-1)
 - Per-joint limits (deg) and optional inversion
 - Sends degrees CSV "d1,d2,...,dN\n" by default (Arduino expects degrees)
 - Watchdog: if joint_states stop arriving, send neutral pose or stop pulses
 - Optional smoothing (simple exponential)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import serial
import threading
import time
import math
from typing import Dict, List

DEFAULT_PORT = '/dev/ttyACM0'
DEFAULT_BAUD = 115200
PUB_RATE_HZ = 50.0            # how often we write to serial (and expected incoming joint update rate)
WATCHDOG_TIMEOUT = 0.25       # seconds - fail-safe if no messages
SMOOTH_ALPHA = 0.4            # 0 => no smoothing; closer to 1 => faster changes

# Example mapping: replace with your URDF joint names in the order of servo channels
# for 12 servos:
DEFAULT_JOINT_ORDER = [
  'fl_coxa_joint','fl_femur_joint','fl_tibia_joint',
  'fr_coxa_joint','fr_femur_joint','fr_tibia_joint',
  'rl_coxa_joint','rl_femur_joint','rl_tibia_joint',
  'rr_coxa_joint','rr_femur_joint','rr_tibia_joint'
]


# Per-joint configuration: min_deg, max_deg, neutral_deg, invert(bool)
# By default we use -45..+45 safe range and neutral 0.
def make_default_limits(n):
    return [{"min": -90.0, "max": 90.0, "neutral": 0.0, "invert": False} for _ in range(n)]

class SerialBridge(Node):
    def __init__(self):
        super().__init__('serial_bridge_jointstate')
        self.declare_parameter('port', DEFAULT_PORT)
        self.declare_parameter('baud', DEFAULT_BAUD)
        self.declare_parameter('topic', '/joint_states')
        self.declare_parameter('joint_order', DEFAULT_JOINT_ORDER)
        self.declare_parameter('limits', None)  # optional override json-like (see README)
        self.declare_parameter('watchdog_timeout', WATCHDOG_TIMEOUT)
        self.declare_parameter('smooth_alpha', SMOOTH_ALPHA)
        self.declare_parameter('publish_rate', PUB_RATE_HZ)
        self.declare_parameter('send_in_degrees', True)  # else send radians

        port = self.get_parameter('port').value
        baud = int(self.get_parameter('baud').value)
        self.topic = self.get_parameter('topic').value
        self.joint_order: List[str] = list(self.get_parameter('joint_order').value)
        limits_param = self.get_parameter('limits').value
        if limits_param:
            # expected to be a list-of-dicts with keys min,max,neutral,invert
            self.limits = limits_param
        else:
            self.limits = make_default_limits(len(self.joint_order))

        self.watchdog_timeout = float(self.get_parameter('watchdog_timeout').value)
        self.alpha = float(self.get_parameter('smooth_alpha').value)
        self.pub_rate = float(self.get_parameter('publish_rate').value)
        self.send_in_degrees = bool(self.get_parameter('send_in_degrees').value)

        self.get_logger().info(f'Opening serial {port} @ {baud}')
        try:
            self.ser = serial.Serial(port, baud, timeout=0.01)
        except Exception as e:
            self.get_logger().error(f'Failed to open serial {port}: {e}')
            raise

        self.lock = threading.Lock()
        self.last_msg_time = 0.0

        # latest commanded positions (radians) in joint_order
        self.target = [math.radians(l["neutral"]) for l in self.limits]
        # smoothed output (radians)
        self.smoothed = list(self.target)

        # mapping from incoming JointState names -> position value
        self.latest_from_topic = {}

        self.sub = self.create_subscription(JointState, self.topic, self.cb_joint_state, 10)
        self.timer = self.create_timer(1.0 / self.pub_rate, self.timer_cb)

    def cb_joint_state(self, msg: JointState):
        with self.lock:
            # store incoming values by name
            for i, name in enumerate(msg.name):
                # assume msg.position aligns with name length
                if i < len(msg.position):
                    self.latest_from_topic[name] = float(msg.position[i])
            self.last_msg_time = time.time()

    def timer_cb(self):
        now = time.time()
        with self.lock:
            # if no recent joint message -> watchdog action
            if (now - self.last_msg_time) > self.watchdog_timeout:
                # set target to neutral
                self.target = [math.radians(l["neutral"]) for l in self.limits]
            else:
                # build target array according to joint_order
                for idx, jn in enumerate(self.joint_order):
                    if jn in self.latest_from_topic:
                        v = float(self.latest_from_topic[jn])
                        # incoming messages typically in radians; ensure correct unit by checking flag
                        # assuming msg sent in radians (standard) — yes
                        self.target[idx] = v
                    # else keep previous target (or neutral if never set)

            # smoothing
            for i in range(len(self.smoothed)):
                self.smoothed[i] = self.alpha * self.target[i] + (1.0 - self.alpha) * self.smoothed[i]

            # prepare output values
            if self.send_in_degrees:
                out_vals = [math.degrees(self.smoothed[i]) for i in range(len(self.smoothed))]
            else:
                out_vals = [self.smoothed[i] for i in range(len(self.smoothed))]

            # clamp and invert per limits
            out_vals_clamped = []
            for i, v in enumerate(out_vals):
                lim = self.limits[i]
                val = -v if lim.get("invert", False) else v
                val = max(lim["min"], min(lim["max"], val))
                out_vals_clamped.append(val)

            # build CSV line (degrees -> two decimals)
            line = ",".join(f"{x:.2f}" for x in out_vals_clamped) + "\n"
            try:
                self.get_logger().info(f"Serial CSV -> {line.strip()}")
                self.ser.write(line.encode('ascii'))
            except Exception as e:
                self.get_logger().error(f"Serial write failed: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = SerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
