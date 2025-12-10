#!/usr/bin/env python3
import sys
import select
import termios
import tty
import serial

import rclpy
from rclpy.node import Node


# Valid keys and human-readable actions
VALID_KEYS = {
    "w": "Forward",
    "s": "Backward",
    "a": "Left",
    "d": "Right",
    "q": "Yaw Left",
    "e": "Yaw Right",
    "x": "Stop/Home",
    "+": "Faster",
    "-": "Slower",
}


HELP_TEXT = """
Spiderbot Serial Teleop
-----------------------
Keys:
  w : forward
  s : backward
  a : left
  d : right
  q : yaw left  (rotate CCW)
  e : yaw right (rotate CW)
  x : stop / home
  + : faster (smaller step delay)
  - : slower (larger step delay)

CTRL+C to exit.
"""


def get_key(settings, timeout=0.1):
    """
    Non-blocking key reader (similar to teleop_twist_keyboard).
    Returns '' if no key pressed within timeout.
    """
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if rlist else ""
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class SpiderSerialTeleop(Node):
    def __init__(self):
        super().__init__("spider_serial_teleop")

        # Expose port and baud as ROS2 params, but with defaults
        self.declare_parameter("port", "/dev/ttyACM0")
        self.declare_parameter("baud", 115200)

        port = self.get_parameter("port").get_parameter_value().string_value
        baud = self.get_parameter("baud").get_parameter_value().integer_value

        self.get_logger().info(f"Opening serial port {port} @ {baud}...")
        try:
            self.ser = serial.Serial(port, baudrate=baud, timeout=0.05)
        except Exception as e:
            self.get_logger().error(f"Failed to open serial port: {e}")
            raise

        # Optional: small delay to allow Arduino to reset after opening serial
        self.get_logger().info("Waiting for Arduino to reset...")
        self.create_timer(2.0, self._print_ready_once)
        self._ready_printed = False

    def _print_ready_once(self):
        if not self._ready_printed:
            self.get_logger().info("Ready for teleop. Press keys as shown above.")
            self._ready_printed = True

    def send_key(self, key: str):
        """Send a single character to the Arduino if it's valid."""
        if key in VALID_KEYS:
            try:
                self.ser.write(key.encode("utf-8"))
                self.ser.flush()
                self.get_logger().info(f"Sent '{key}' -> {VALID_KEYS[key]}")
            except Exception as e:
                self.get_logger().error(f"Error writing to serial: {e}")
        elif key == '\x03':  # CTRL+C
            raise KeyboardInterrupt
        else:
            # Ignore unknown keys; could also log as debug
            self.get_logger().debug(f"Ignored key: {repr(key)}")

    def destroy_node(self):
        try:
            if hasattr(self, "ser") and self.ser.is_open:
                self.get_logger().info("Closing serial port...")
                self.ser.close()
        except Exception as e:
            self.get_logger().warn(f"Error closing serial port: {e}")
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    settings = termios.tcgetattr(sys.stdin)

    node = None
    try:
        node = SpiderSerialTeleop()
        print(HELP_TEXT)

        # Main loop: read key, send to Arduino, spin ROS briefly
        while rclpy.ok():
            key = get_key(settings, timeout=0.1)
            if key:
                node.send_key(key)

            # We don't rely on ROS timers heavily here, but keep spin_once
            rclpy.spin_once(node, timeout_sec=0.0)

    except KeyboardInterrupt:
        print("\nExiting teleop.")

    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)


if __name__ == "__main__":
    main()
