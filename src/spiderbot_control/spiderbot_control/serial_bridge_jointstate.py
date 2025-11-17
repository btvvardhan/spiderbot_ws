#!/usr/bin/env python3
"""
Improved Serial Bridge for Spiderbot
- Clean CSV transmission (no garbage)
- Comprehensive logging
- No buffering/waiting
- Persistent angles
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import serial
import time
import math
import sys

# Joint order matching robot URDF
DEFAULT_JOINT_ORDER = [
    'fl_coxa_joint', 'fl_femur_joint', 'fl_tibia_joint',  # Front Left
    'rl_coxa_joint', 'rl_femur_joint', 'rl_tibia_joint',  # Rear Left
    'rr_coxa_joint', 'rr_femur_joint', 'rr_tibia_joint',  # Rear Right
    'fr_coxa_joint', 'fr_femur_joint', 'fr_tibia_joint',  # Front Right
]

class SerialBridge(Node):
    def __init__(self):
        super().__init__('serial_bridge')
        
        # Serial port configuration
        self.declare_parameter('port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('update_rate', 20.0)  # Hz
        
        port = self.get_parameter('port').value
        baud = self.get_parameter('baudrate').value
        rate = self.get_parameter('update_rate').value
        
        self.get_logger().info(f'🔌 Opening {port} @ {baud} baud...')
        
        try:
            # Open serial with explicit settings
            self.ser = serial.Serial(
                port=port,
                baudrate=baud,
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE,
                timeout=0.1,
                write_timeout=0.5,
                xonxoff=False,
                rtscts=False,
                dsrdtr=False
            )
        except serial.SerialException as e:
            self.get_logger().error(f'❌ Failed to open serial port: {e}')
            sys.exit(1)
        
        # Wait for Arduino reset and initialization
        self.get_logger().info('⏳ Waiting for Arduino reset...')
        time.sleep(3.0)
        
        # Clear any garbage from buffers
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
        # Wait for READY signal
        ready = False
        start_time = time.time()
        while not ready and (time.time() - start_time < 5.0):
            if self.ser.in_waiting:
                line = self.ser.readline().decode('ascii', errors='ignore').strip()
                if line == 'READY':
                    ready = True
                    self.get_logger().info(f'✅ Arduino ready: {line}')
        
        if not ready:
            self.get_logger().warn('⚠️  No READY signal received, continuing anyway...')
        
        # State tracking
        self.latest_positions = {}
        self.last_sent_line = ""
        self.send_count = 0
        self.skip_count = 0
        
        # Initialize with neutral positions
        for joint in DEFAULT_JOINT_ORDER:
            self.latest_positions[joint] = 0.0  # radians (90 degrees after conversion)
        
        # ROS2 subscription
        self.sub = self.create_subscription(
            JointState, 
            '/joint_states', 
            self.joint_state_callback, 
            10
        )
        
        # Timer for sending (no buffering - sends latest only)
        update_period = 1.0 / rate
        self.timer = self.create_timer(update_period, self.send_angles)
        
        self.get_logger().info(f'✅ Ready to transmit at {rate} Hz')

    def joint_state_callback(self, msg: JointState):
        """Update latest joint positions from ROS2 topic"""
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.latest_positions[name] = float(msg.position[i])

    def send_angles(self):
        """Send current angles to Arduino (no queuing, latest only)"""
        # Build angle list in correct order
        angles_deg = []
        
        for joint_name in DEFAULT_JOINT_ORDER:
            # Get position in radians (default to 0.0 if not available)
            rad = self.latest_positions.get(joint_name, 0.0)
            
            # Convert to degrees and add 90-degree offset
            deg = (rad * 180.0 / math.pi) + 90.0
            
            # Clamp to valid servo range [0, 180]
            deg = max(0.0, min(180.0, deg))
            
            # Round to integer
            deg_int = int(round(deg))
            angles_deg.append(deg_int)
        
        # Create CSV line
        csv_line = ",".join(map(str, angles_deg)) + "\n"
        
        # Skip if identical to last transmission (avoid redundant sends)
        if csv_line == self.last_sent_line:
            self.skip_count += 1
            return
        
        # Transmit to Arduino
        try:
            # Encode and send (ASCII only, no garbage)
            self.ser.write(csv_line.encode('ascii'))
            self.ser.flush()  # Ensure immediate transmission
            
            # Update tracking
            self.last_sent_line = csv_line
            self.send_count += 1
            
            # Log transmission (every 20 sends to avoid spam)
            if self.send_count % 20 == 0:
                self.get_logger().info(
                    f'📤 TX #{self.send_count}: {csv_line.strip()} '
                    f'(skipped {self.skip_count} duplicates)'
                )
                self.skip_count = 0
            
        except serial.SerialTimeoutException:
            self.get_logger().error('❌ Serial write timeout')
        except Exception as e:
            self.get_logger().error(f'❌ Write error: {e}')

    def __del__(self):
        """Cleanup on shutdown"""
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()
            self.get_logger().info('🔌 Serial port closed')


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