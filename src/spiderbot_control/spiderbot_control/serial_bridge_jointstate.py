#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import serial
import time
import math

DEFAULT_JOINT_ORDER = [
  'fl_coxa_joint','fl_femur_joint','fl_tibia_joint',
  'fr_coxa_joint','fr_femur_joint','fr_tibia_joint',
  'rl_coxa_joint','rl_femur_joint','rl_tibia_joint',
  'rr_coxa_joint','rr_femur_joint','rr_tibia_joint'
]

class SerialBridge(Node):
    def __init__(self):
        super().__init__('serial_bridge')
        
        port = '/dev/ttyACM0'
        baud = 115200
        
        self.get_logger().info(f'Opening {port} @ {baud}...')
        
        # Open with proper settings
        self.ser = serial.Serial(
            port=port,
            baudrate=baud,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=0.1,
            write_timeout=0.1,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False
        )
        
        # Wait for Arduino reset
        time.sleep(3.0)
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        
        self.latest_positions = {}
        self.last_sent = ""
        
        self.sub = self.create_subscription(JointState, '/joint_states', self.cb, 10)
        self.timer = self.create_timer(0.05, self.send)  # 20Hz
        
        self.get_logger().info('✓ Ready')

    def cb(self, msg: JointState):
        for i, name in enumerate(msg.name):
            if i < len(msg.position):
                self.latest_positions[name] = float(msg.position[i])

    def send(self):
        vals = []
        for jn in DEFAULT_JOINT_ORDER:
            rad = self.latest_positions.get(jn, 0.0)
            deg = (rad * 180.0 / math.pi) + 90.0
            deg = max(0, min(180, int(deg)))
            vals.append(str(deg))
        
        line = ",".join(vals) + "\n"
        
        # Skip duplicate sends
        if line == self.last_sent:
            return
        
        self.last_sent = line
        
        try:
                        # In your Python bridge, add:
            self.get_logger().info(f'TX: {line.strip()}')
            self.ser.write(line.encode('ascii'))
            self.ser.flush()
        except Exception as e:
            self.get_logger().error(f'Write error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = SerialBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.ser.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()