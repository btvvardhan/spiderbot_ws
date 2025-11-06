# probe_map.py
import time, numpy as np
from rclpy.node import Node
import rclpy
from std_msgs.msg import Float64MultiArray

NEUTRAL = np.array([ 0.6, 0.5, -0.2,  -0.6, 0.5, -0.2,  0.6, 0.5, -0.2,  -0.6, 0.5, -0.2 ], dtype=float)

class Prober(Node):
    def __init__(self):
        super().__init__('map_prober')
        self.pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 1)
        self.idx = 0
        self.timer = self.create_timer(1.0, self.tick)

    def tick(self):
        arr = NEUTRAL.copy()
        # small “flash” on current index
        arr[self.idx] += 0.15
        self.pub.publish(Float64MultiArray(data=arr.tolist()))
        self.get_logger().info(f'FLASH index {self.idx}')
        self.idx = (self.idx + 1) % 12

def main():
    rclpy.init()
    n = Prober()
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    n.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
