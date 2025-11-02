#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import curses
import time
from typing import Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


KEY_ESC = 27

class TeleopKeyboard(Node):
    """
    Keyboard teleop for Spiderbot (WASD + QE + SPACE + R + ↑/↓), publishing /cmd_vel.

    Keys:
      W/S : forward/back (x)
      A/D : left/right (y)
      Q/E : rotate left/right (yaw)
      SPACE : emergency stop (zero all)
      R : reset to zero
      ↑ / ↓ : increase/decrease velocity scale (0.1x .. 2.0x)
      ESC : exit
    """

    def __init__(self):
        super().__init__("teleop_keyboard")

        # Params (you can override with --ros-args -p name:=value)
        self.declare_parameter("cmd_topic", "/cmd_vel")
        self.declare_parameter("rate_hz", 30.0)
        self.declare_parameter("max_lin_vel", 1.0)
        self.declare_parameter("max_ang_vel", 1.0)
        self.declare_parameter("step", 0.15)     # per key repeat/frame
        self.declare_parameter("scale", 1.0)     # velocity multiplier

        self.cmd_topic = self.get_parameter("cmd_topic").get_parameter_value().string_value
        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.max_lin = float(self.get_parameter("max_lin_vel").value)
        self.max_ang = float(self.get_parameter("max_ang_vel").value)
        self.step = float(self.get_parameter("step").value)
        self.scale = float(self.get_parameter("scale").value)

        self.pub = self.create_publisher(Twist, self.cmd_topic, 10)

        # current command [vx, vy, wz]
        self.vx = 0.0
        self.vy = 0.0
        self.wz = 0.0

        self.get_logger().info(
            f"Keyboard teleop ready -> {self.cmd_topic} | "
            f"rate={self.rate_hz} Hz, max_lin={self.max_lin}, max_ang={self.max_ang}, step={self.step}, scale={self.scale}"
        )

    def _clamp(self):
        # scale then clamp
        sx = max(min(self.vx * self.scale, self.max_lin), -self.max_lin)
        sy = max(min(self.vy * self.scale, self.max_lin), -self.max_lin)
        sz = max(min(self.wz * self.scale, self.max_ang), -self.max_ang)
        return sx, sy, sz

    def _publish(self):
        sx, sy, sz = self._clamp()
        msg = Twist()
        msg.linear.x = float(sx)
        msg.linear.y = float(sy)
        msg.angular.z = float(sz)
        self.pub.publish(msg)

    def _draw_help(self, stdscr):
        lines = [
            "SPIDERBOT KEYBOARD TELEOP",
            "--------------------------",
            "W/S : Forward/Back",
            "A/D : Strafe Left/Right",
            "Q/E : Rotate Left/Right (yaw)",
            "SPACE: Emergency stop (zero)",
            "R    : Reset to zero",
            "UP/DOWN: +/− velocity scale",
            "ESC  : Exit",
            "",
            f"scale={self.scale:.2f}  vx={self.vx:.2f}  vy={self.vy:.2f}  wz={self.wz:.2f}",
        ]
        stdscr.erase()
        for i, ln in enumerate(lines):
            stdscr.addstr(i, 0, ln)
        stdscr.refresh()

    def run_curses(self, stdscr):
        # make input non-blocking; let OS key-repeat do the “held key” effect
        stdscr.nodelay(True)
        curses.curs_set(0)

        period = 1.0 / max(1.0, self.rate_hz)
        last_print = 0.0

        self._draw_help(stdscr)
        while rclpy.ok():
            t0 = time.time()
            # Read all available keypresses this frame
            while True:
                try:
                    ch = stdscr.getch()
                except KeyboardInterrupt:
                    return
                if ch == -1:
                    break

                # Movement
                if ch in (ord('w'), ord('W')):
                    self.vx += self.step
                elif ch in (ord('s'), ord('S')):
                    self.vx -= self.step
                elif ch in (ord('a'), ord('A')):
                    self.vy += self.step   # left positive (match your ISAAC script)
                elif ch in (ord('d'), ord('D')):
                    self.vy -= self.step   # right negative
                elif ch in (ord('q'), ord('Q')):
                    self.wz += self.step
                elif ch in (ord('e'), ord('E')):
                    self.wz -= self.step

                # Scale
                elif ch == curses.KEY_UP:
                    self.scale = min(2.0, self.scale + 0.1)
                elif ch == curses.KEY_DOWN:
                    self.scale = max(0.1, self.scale - 0.1)

                # Stop / reset
                elif ch == ord(' '):
                    self.vx = self.vy = self.wz = 0.0
                elif ch in (ord('r'), ord('R')):
                    self.vx = self.vy = self.wz = 0.0

                # Exit
                elif ch == KEY_ESC:
                    return

            # Publish current command (defaults to zero if you never press a key)
            self._publish()

            # occasional HUD refresh
            now = time.time()
            if now - last_print > 0.1:
                self._draw_help(stdscr)
                last_print = now

            # allow ROS to process parameter updates (optional)
            rclpy.spin_once(self, timeout_sec=0.0)

            # sleep to target rate
            dt = time.time() - t0
            if period - dt > 0:
                time.sleep(period - dt)


def main():
    rclpy.init()
    node = TeleopKeyboard()
    try:
        curses.wrapper(node.run_curses)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
