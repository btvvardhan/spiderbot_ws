#!/usr/bin/env python3
"""
Raspberry Pi Bridge Node
- Subscribes to /joint_states from main computer
- Sends servo angles to Arduino
- Reads IMU data from Arduino and publishes to /imu
- Republishes LIDAR data from /scan to network
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Imu, LaserScan, MagneticField
from std_msgs.msg import Header
from geometry_msgs.msg import Quaternion, Vector3
import serial
import math
import time
import threading

class PiBridgeNode(Node):
    def __init__(self):
        super().__init__('pi_bridge_node')
        
        # Parameters
        self.declare_parameter('arduino_port', '/dev/ttyACM0')
        self.declare_parameter('baudrate', 115200)
        self.declare_parameter('deg_offset', 90.0)
        self.declare_parameter('servo_min_deg', 0.0)
        self.declare_parameter('servo_max_deg', 180.0)
        
        port = self.get_parameter('arduino_port').value
        baud = int(self.get_parameter('baudrate').value)
        self.deg_offset = float(self.get_parameter('deg_offset').value)
        self.servo_min = float(self.get_parameter('servo_min_deg').value)
        self.servo_max = float(self.get_parameter('servo_max_deg').value)
        
        # Serial connection to Arduino
        self.get_logger().info(f'🔌 Connecting to Arduino on {port}...')
        try:
            self.ser = serial.Serial(
                port=port,
                baudrate=baud,
                timeout=0.1,
                write_timeout=0.5
            )
            time.sleep(2.0)  # Wait for Arduino reset
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
            self.get_logger().info('✅ Arduino connected')
        except serial.SerialException as e:
            self.get_logger().error(f'❌ Failed to open Arduino: {e}')
            raise
        
        # Subscribe to joint commands from main computer
        self.joint_sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )
        
        # Publish IMU data
        self.imu_pub = self.create_publisher(Imu, '/imu', 10)
        self.mag_pub = self.create_publisher(MagneticField, '/mag', 10)
        
        # Republish LIDAR (pass-through for network visibility)
        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )
        self.scan_pub = self.create_publisher(LaserScan, '/scan_republished', 10)
        
        # State tracking
        self.last_sent_line = ""
        self.send_count = 0
        self.skip_count = 0
        
        # Start Arduino reader thread
        self.running = True
        self.reader_thread = threading.Thread(target=self.read_arduino)
        self.reader_thread.daemon = True
        self.reader_thread.start()
        
        self.get_logger().info('🤖 Pi Bridge Ready!')
        self.get_logger().info('   📥 Subscribing: /joint_states')
        self.get_logger().info('   📤 Publishing: /imu, /mag, /scan_republished')

    def joint_callback(self, msg: JointState):
        """Receive joint commands from main computer, send to Arduino"""
        angles_deg = []
        for pos in msg.position:
            deg = (pos * 180.0 / math.pi) + self.deg_offset
            deg = max(self.servo_min, min(self.servo_max, deg))
            angles_deg.append(int(round(deg)))
        
        csv_line = ",".join(map(str, angles_deg)) + "\n"
        
        # Skip duplicates
        if csv_line == self.last_sent_line:
            self.skip_count += 1
            return
        
        try:
            self.ser.write(csv_line.encode('ascii'))
            self.ser.flush()
            self.last_sent_line = csv_line
            self.send_count += 1
            
            if self.send_count % 50 == 0:
                self.get_logger().info(
                    f'📤 Servos: #{self.send_count} (skipped {self.skip_count} dups)'
                )
                self.skip_count = 0
                
        except Exception as e:
            self.get_logger().error(f'❌ Serial write error: {e}')

    def read_arduino(self):
        """Background thread to read IMU data from Arduino"""
        while self.running and rclpy.ok():
            try:
                if self.ser.in_waiting:
                    line = self.ser.readline().decode('ascii', errors='ignore').strip()
                    
                    # Parse IMU data
                    if line.startswith('IMU,'):
                        self.parse_imu_data(line)
                    elif line == "READY":
                        self.get_logger().info('✅ Arduino is READY')
                        
            except Exception as e:
                self.get_logger().error(f'❌ Arduino read error: {e}')
                time.sleep(0.1)

    def parse_imu_data(self, line: str):
        """Parse IMU data: IMU,ax,ay,az,gx,gy,gz,mx,my,mz"""
        try:
            parts = line.split(',')
            if len(parts) < 10:
                return
            
            # Parse values
            ax = float(parts[1])  # m/s²
            ay = float(parts[2])
            az = float(parts[3])
            gx = float(parts[4])  # rad/s
            gy = float(parts[5])
            gz = float(parts[6])
            mx = float(parts[7])  # raw magnetometer
            my = float(parts[8])
            mz = float(parts[9])
            
            # Publish IMU message
            imu_msg = Imu()
            imu_msg.header = Header()
            imu_msg.header.stamp = self.get_clock().now().to_msg()
            imu_msg.header.frame_id = 'imu_link'
            
            # Linear acceleration
            imu_msg.linear_acceleration.x = ax
            imu_msg.linear_acceleration.y = ay
            imu_msg.linear_acceleration.z = az
            
            # Angular velocity
            imu_msg.angular_velocity.x = gx
            imu_msg.angular_velocity.y = gy
            imu_msg.angular_velocity.z = gz
            
            # Orientation unknown (set covariance to -1)
            imu_msg.orientation_covariance[0] = -1.0
            imu_msg.linear_acceleration_covariance[0] = 0.01
            imu_msg.angular_velocity_covariance[0] = 0.01
            
            self.imu_pub.publish(imu_msg)
            
            # Publish magnetometer separately
            mag_msg = MagneticField()
            mag_msg.header = imu_msg.header
            mag_msg.magnetic_field.x = mx
            mag_msg.magnetic_field.y = my
            mag_msg.magnetic_field.z = mz
            mag_msg.magnetic_field_covariance[0] = 0.0
            
            self.mag_pub.publish(mag_msg)
            
        except (ValueError, IndexError) as e:
            self.get_logger().warn(f'⚠️  Failed to parse IMU: {e}')

    def scan_callback(self, msg: LaserScan):
        """Republish LIDAR data for network visibility"""
        self.scan_pub.publish(msg)

    def __del__(self):
        """Cleanup"""
        self.running = False
        if hasattr(self, 'ser') and self.ser.is_open:
            self.ser.close()


def main():
    rclpy.init()
    node = PiBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('🛑 Shutting down...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()