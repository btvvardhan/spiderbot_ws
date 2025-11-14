#!/usr/bin/env python3
import os, math, time, xml.etree.ElementTree as ET
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

PKG = 'spiderbot_description'
URDF_FILE = 'spidy.urdf'

def parse_joint_names(urdf_path):
    root = ET.parse(urdf_path).getroot()
    names = []
    for j in root.findall('joint'):
        t = (j.get('type') or '').lower()
        if t in ('revolute', 'continuous', 'prismatic'):
            names.append(j.get('name'))
    return names

class Animator(Node):
    def __init__(self, joints, hz=50.0, freq=0.6, amp=0.35):
        super().__init__('spiderbot_animator')
        self.pub = self.create_publisher(JointState, '/anim_joint_states', 10)
        self.joints = joints
        self.dt = 1.0 / hz
        self.w = 2*math.pi*freq
        self.amp = amp
        self.t0 = time.time()
        self.timer = self.create_timer(self.dt, self.tick)
        self.get_logger().info(f'Animating {len(joints)} joints @ {hz} Hz, freq={freq} Hz, amp={amp} rad')

    def tick(self):
        t = time.time() - self.t0
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.joints
        msg.position = [self.amp*math.sin(self.w*t + i*math.pi/6.0) for i in range(len(self.joints))]
        self.pub.publish(msg)

def main():
    rclpy.init()
    urdf = os.path.join(get_package_share_directory(PKG), 'urdf', URDF_FILE)
    joints = parse_joint_names(urdf)
    if not joints:
        print('No movable joints found in URDF.')
        return
    node = Animator(joints)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
