#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Serial Bridge for Spiderbot

- Subscribes to /joint_states (policy order: [FL, FR, RL, RR]).
- Transmits to Arduino in configurable hardware order (default: [FL, RL, RR, FR]).
- Converts rad -> deg (+deg_offset), clamps, dedups.

Params:
  port: /dev/ttyACM0
  baudrate: 115200
  update_rate: 20.0
  deg_offset: 90.0
  servo_min_deg: 0.0
  servo_max_deg: 180.0
  listen_order:  list of joint names (default FL,FR,RL,RR)
  arduino_order: list of joint names (default FL,RL,RR,FR)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import serial
import time
import math
import sys

# Joint name groups
FL = ["fl_coxa_joint", "fl_femur_joint", "fl_tibia_joint"]
FR = ["fr_coxa_joint", "fr_femur_joint", "fr_tibia_joint"]
RL = ["rl_coxa_joint", "rl_femur_joint", "rl_tibia_joint"]
RR = ["rr_coxa_joint", "rr_femur_joint", "rr_tibia_joint"]

LISTEN_ORDER_DEFAULT = FL + FR + RL + RR           # from policy_cpg_node
ARDUINO_ORDER_DEFAULT = FL + RL + RR + FR          # your hardware order


class SerialBridge(Node):
    def __init__(self):
        super().__init__('serial_bridge')

        # Parameters
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('update_rate', 20.0)
        self.declare_parameter('deg_offset', 90.0)
        self.declare_parameter('servo_min_deg', 0.0)
        self.declare_parameter('servo_max_deg', 180.0)
        self.declare_parameter('listen_order', LISTEN_ORDER_DEFAULT)
        self.declare_parameter('arduino_order', ARDUINO_ORDER_DEFAULT)

        port = self.get_parameter('port').value
        baud = int(self.get_parameter('baudrate').value)
        rate = float(self.get_parameter('update_rate').value)
        self.deg_offset = float(self.get_parameter('deg_offset').value)
        self.servo_min = float(self.get_parameter('servo_min_deg').value)
        self.servo_max = float(self.get_parameter('servo_max_deg').value)

        self.listen_order = list(self.get_parameter('listen_order').value)
        self.arduino_order = list(self.get_parameter('arduino_order').value)

        self.get_logger().info(
            "🛰️  SerialBridge starting\n"
            f"   Listen order (from /joint_states): {self.listen_order}\n"
            f"   Arduino TX order:                  {self.arduino_order}"
        )

        # Serial init
        self.get_logger().info(f'🔌 Opening {port} @ {baud} baud...')
        try:
            self.ser = serial.Serial(
                port=port, baudrate=baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1, write_timeout=0.5,
                xonxoff=False, rtscts=False, dsrdtr=False
            )
        except serial.SerialException as e:
            self.get_logger().error(f'❌ Failed to open serial port: {e}')
            sys.exit(1)

        self.get_logger().info('⏳ Waiting for Arduino reset...')
        time.sleep(3.0)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

        # Optional handshake
        ready = False
        start_time = time.time()
        while not ready and (time.time() - start_time < 5.0):
            if self.ser.in_waiting:
                line = self.ser.readline().decode('ascii', errors='ignore').strip()
                if line == 'READY':
                    ready = True
                    self.get_logger().info('✅ Arduino ready')
        if not ready:
            self.get_logger().warn('⚠️  No READY signal, continuing...')

        # State
        all_names = set(self.listen_order) | set(self.arduino_order)
        self.latest_positions = {name: 0.0 for name in all_names}
        self.last_sent_line = ""
        self.send_count = 0
        self.skip_count = 0

        # ROS I/O
        self.sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.timer = self.create_timer(1.0 / rate, self.send_angles)
        self.get_logger().info(f'✅ Ready: publishing to Arduino at {rate:.1f} Hz')

    def joint_state_callback(self, msg: JointState):
        n = min(len(msg.name), len(msg.position))
        for i in range(n):
            self.latest_positions[msg.name[i]] = float(msg.position[i])

    def send_angles(self):
        angles_deg = []
        for joint_name in self.arduino_order:
            rad = float(self.latest_positions.get(joint_name, 0.0))
            deg = (rad * 180.0 / math.pi) + self.deg_offset
            deg = max(self.servo_min, min(self.servo_max, deg))
            angles_deg.append(int(round(deg)))

        csv_line = ",".join(map(str, angles_deg)) + "\n"
        if csv_line == self.last_sent_line:
            self.skip_count += 1
            return

        try:
            self.ser.write(csv_line.encode('ascii'))
            self.ser.flush()
            self.last_sent_line = csv_line
            self.send_count += 1
            if self.send_count % 20 == 0:
                self.get_logger().info(
                    f'📤 TX #{self.send_count}: {csv_line.strip()} (skipped {self.skip_count} dups)'
                )
                self.skip_count = 0
        except serial.SerialTimeoutException:
            self.get_logger().error('❌ Serial write timeout')
        except Exception as e:
            self.get_logger().error(f'❌ Serial write error: {e}')

    def __del__(self):
        try:
            if hasattr(self, 'ser') and self.ser.is_open:
                self.ser.close()
                self.get_logger().info('🔌 Serial port closed')
        except Exception:
            pass


def main(args=None):
    rclpy.init(args=args)
    try:
        node = SerialBridge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if 'node' in locals():
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
