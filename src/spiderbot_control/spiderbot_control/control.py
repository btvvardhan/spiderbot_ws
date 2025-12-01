#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spiderbot Teleop - Publishes ASCII chars to a ROS2 topic 'cmd'

Sends: 'w', 's', 'a', 'd', 'q', 'e', 'x', '+', '-' as std_msgs/String
"""

import curses
import time

from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

KEY_ESC = 27


class SpiderbotTeleop(Node):
    def __init__(self):
        super().__init__("control")

        # Parameters
        self.declare_parameter("cmd_topic", "cmd")
        self.declare_parameter("debug", False)

        self.cmd_topic = self.get_parameter("cmd_topic").get_parameter_value().string_value
        self.debug = self.get_parameter("debug").get_parameter_value().bool_value

        # Publisher: sends single-character commands
        self.cmd_pub = self.create_publisher(String, self.cmd_topic, 10)

        # State
        self.current_gait: str = 'x'      # Start stopped
        self.last_sent_cmd: Optional[str] = None

        self.get_logger().info(f"🕷️ Spiderbot Teleop Ready! Publishing to '{self.cmd_topic}'")

    def _send_cmd(self, cmd: str):
        """Publish single-character command on topic 'cmd'"""
        # Only send if changed (no spam) — except speed commands
        if cmd in ['w', 's', 'a', 'd', 'q', 'e', 'x'] and cmd == self.last_sent_cmd:
            return

        msg = String()
        msg.data = cmd + "\n"
        self.cmd_pub.publish(msg)
        self.last_sent_cmd = cmd

        names = {
            'w': 'FORWARD', 's': 'BACKWARD', 'a': 'LEFT',
            'd': 'RIGHT', 'q': 'YAW_LEFT', 'e': 'YAW_RIGHT',
            'x': 'STOP', '+': 'FASTER', '-': 'SLOWER'
        }

        if self.debug:
            self.get_logger().info(f"📤 CMD: '{cmd}' = {names.get(cmd, cmd)}")
        else:
            self.get_logger().info(f"→ {names.get(cmd, cmd)}")



    def _draw_help(self, stdscr):
        """Draw UI"""
        lines = [
            "╔═══════════════════════════════════════════╗",
            "║   SPIDERBOT TELEOP (ROS2 topic 'cmd')    ║",
            "╚═══════════════════════════════════════════╝",
            "",
            "Movement:",
            "  W / S  : Forward / Backward",
            "  A / D  : Strafe Left / Right",
            "  Q / E  : Yaw Left / Right",
            "  X      : STOP",
            "",
            "Speed:",
            "  + / -  : Faster / Slower",
            "",
            "Exit:",
            "  ESC    : Quit",
            "",
            "─────────────────────────────────────────────",
            f"Gait: {self.current_gait.upper()}",
            f"Topic: {self.cmd_topic}",
            "Status: ✓ Publishing",
            "─────────────────────────────────────────────",
        ]

        stdscr.erase()
        for i, ln in enumerate(lines):
            try:
                stdscr.addstr(i, 0, ln)
            except Exception:
                pass
        stdscr.refresh()

    def run_curses(self, stdscr):
        """Main loop"""
        stdscr.nodelay(True)
        curses.curs_set(0)

        last_draw = 0.0
        self._draw_help(stdscr)

        while rclpy.ok():
            try:
                ch = stdscr.getch()
            except KeyboardInterrupt:
                break

            new_gait = self.current_gait

            if ch != -1:
                # Map keys to commands
                if ch in (ord('w'), ord('W')):
                    new_gait = 'w'
                elif ch in (ord('s'), ord('S')):
                    new_gait = 's'
                elif ch in (ord('a'), ord('A')):
                    new_gait = 'a'
                elif ch in (ord('d'), ord('D')):
                    new_gait = 'd'
                elif ch in (ord('q'), ord('Q')):
                    new_gait = 'q'
                elif ch in (ord('e'), ord('E')):
                    new_gait = 'e'
                elif ch in (ord('x'), ord('X')):
                    new_gait = 'x'
                elif ch == ord('+'):
                    self._send_cmd('+')
                elif ch == ord('-'):
                    self._send_cmd('-')
                elif ch == KEY_ESC:
                    break

                # Send if gait changed
                if new_gait != self.current_gait:
                    self.current_gait = new_gait
                    self._send_cmd(self.current_gait)

            # Redraw UI at ~10 Hz
            now = time.time()
            if now - last_draw > 0.1:
                self._draw_help(stdscr)
                last_draw = now

            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(0.01)


def main():
    rclpy.init()
    node = SpiderbotTeleop()

    try:
        curses.wrapper(node.run_curses)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
