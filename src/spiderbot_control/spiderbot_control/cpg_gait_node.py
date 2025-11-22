#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Float64MultiArray
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Twist
import numpy as np
from .spidy_ik import SpiderIK

class SpiderController(Node):
    def __init__(self):
        super().__init__('spider_controller')
        self.ik = SpiderIK()

        self.declare_parameter("out_topic", "/position_controller/commands")
        out_topic = self.get_parameter("out_topic").get_parameter_value().string_value

        self.joint_names = [
            'fl_coxa_joint','fl_femur_joint','fl_tibia_joint',
            'fr_coxa_joint','fr_femur_joint','fr_tibia_joint',
            'rl_coxa_joint','rl_femur_joint','rl_tibia_joint',
            'rr_coxa_joint','rr_femur_joint','rr_tibia_joint'
        ]

        # With axis-aware IK, you should NOT need sign hacks.
        self.signs = {name: 1.0 for name in self.joint_names}
        self.zeros = {name: 0.0 for name in self.joint_names}

        self.joint_pub = self.create_publisher(JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.pos_pub   = self.create_publisher(Float64MultiArray, out_topic, 10)
        self.cmd_sub   = self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_cb, 10)

        self.current_phase = 0.0
        self.body_height   = 0.10
        self.step_height   = 0.02
        self.stride_length = 0.06
        self.gait_frequency = 1.0
        self.velocity_cmd = np.array([0.0, 0.0, 0.0])

        self.create_timer(1.0/30.0, self.gait_cb)

        self.get_logger().info('Spider Controller initialized')
        self.get_logger().info(f'Publishing trajectory to: {self.joint_pub.topic_name}')
        self.get_logger().info(f'Publishing positions to: {self.pos_pub.topic_name}')

        self.set_standing_pose()

    def cmd_vel_cb(self, msg: Twist):
        self.velocity_cmd[:] = [msg.linear.x, msg.linear.y, msg.angular.z]
        self.get_logger().info(f'Velocity command: vx={msg.linear.x:.2f}, vy={msg.linear.y:.2f}, w={msg.angular.z:.2f}')

    def _apply(self, angles: dict) -> dict:
        out = {}
        for n in self.joint_names:
            out[n] = self.signs[n]*angles.get(n, 0.0) + self.zeros[n]
        return out

    def publish(self, angles: dict, duration_sec: float):
        angles = self._apply(angles)

        traj = JointTrajectory()
        traj.header.stamp = self.get_clock().now().to_msg()
        traj.joint_names = self.joint_names
        pt = JointTrajectoryPoint()
        pt.positions = [angles[n] for n in self.joint_names]
        pt.time_from_start = Duration(sec=int(duration_sec), nanosec=int((duration_sec%1)*1e9))
        traj.points = [pt]
        self.joint_pub.publish(traj)

        msg = Float64MultiArray()
        msg.data = [angles[n] for n in self.joint_names]
        self.pos_pub.publish(msg)

    def set_standing_pose(self):
        ang = self.ik.get_standing_pose(body_height=self.body_height)
        self.publish(ang, duration_sec=2.0)
        self.get_logger().info('Set standing pose')

    def gait_cb(self):
        dt = 1.0/30.0
        self.current_phase = (self.current_phase + 2*np.pi*self.gait_frequency*dt) % (2*np.pi)
        if np.linalg.norm(self.velocity_cmd) < 0.01:
            return

        feet = self._walk_traj(self.current_phase, self.velocity_cmd)
        out = {}
        for leg, p in feet.items():
            sol = self.ik.solve_leg_ik(leg, p)
            if sol is None: 
                continue
            c,f,t = sol
            out[f"{leg}_coxa_joint"]=c
            out[f"{leg}_femur_joint"]=f
            out[f"{leg}_tibia_joint"]=t
        if out:
            self.publish(out, duration_sec=dt)

    def _walk_traj(self, phase: float, vel: np.ndarray) -> dict:
        vx, vy, wz = vel
        stance = {
            'fl': np.array([ 0.18, -0.18, -self.body_height]),
            'fr': np.array([-0.18, -0.18, -self.body_height]),
            'rl': np.array([ 0.18,  0.18, -self.body_height]),
            'rr': np.array([-0.18,  0.18, -self.body_height]),
        }
        speed = np.linalg.norm([vx, vy])
        stride = self.stride_length * np.clip(speed/0.30, 0.0, 1.0)

        feet = {}
        for leg, base in stance.items():
            leg_phase = (phase if leg in ('fl','rr') else phase+np.pi)%(2*np.pi)
            if leg_phase < np.pi:
                s = leg_phase/np.pi
                z = self.step_height*np.sin(s*np.pi)
                x = -stride/2 + s*stride
                feet[leg] = base + np.array([x, 0.0, z])
            else:
                s = (leg_phase-np.pi)/np.pi
                x =  stride/2 - s*stride
                feet[leg] = base + np.array([x, 0.0, 0.0])
            if abs(wz) > 1e-6:
                feet[leg][1] += 0.02*wz
        return feet

def main(args=None):
    rclpy.init(args=args)
    node = SpiderController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
