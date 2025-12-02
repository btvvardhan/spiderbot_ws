#!/usr/bin/env python3
"""
VLM Dashboard Node - Web interface for robot control

Provides:
- Live camera feed
- Scene description display
- Chat interface for robot commands
- Status monitoring

Run: ros2 run spiderbot_vlm vlm_dashboard_node
Access: http://localhost:5000
"""

import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
import threading
from flask import Flask, render_template, Response, request, jsonify
from flask_cors import CORS
import time


class VLMDashboardNode(Node):
    def __init__(self):
        super().__init__('vlm_dashboard_node')
        
        # Parameters
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('description_topic', '/scene_description')
        self.declare_parameter('goal_topic', '/vlm_goal')
        self.declare_parameter('port', 5000)
        
        camera_topic = self.get_parameter('camera_topic').value
        self.description_topic = self.get_parameter('description_topic').value
        self.goal_topic = self.get_parameter('goal_topic').value
        self.port = self.get_parameter('port').value
        
        # State
        self.bridge = CvBridge()
        self.latest_frame = None
        self.latest_description = "No description yet. Click 'Describe Scene' to start."
        self.frame_lock = threading.Lock()
        
        # ROS subscriptions
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )
        
        self.description_sub = self.create_subscription(
            String,
            self.description_topic,
            self.description_callback,
            10
        )
        
        # ROS publishers
        self.goal_pub = self.create_publisher(
            String,
            self.goal_topic,
            10
        )
        
        self.get_logger().info('🎨 VLM Dashboard Node initialized')
        self.get_logger().info(f'   📷 Camera: {camera_topic}')
        self.get_logger().info(f'   🌐 Dashboard: http://localhost:{self.port}')
    
    def image_callback(self, msg: Image):
        """Store latest camera frame"""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            with self.frame_lock:
                self.latest_frame = cv_img
        except Exception as e:
            self.get_logger().warn(f'Failed to convert image: {e}')
    
    def description_callback(self, msg: String):
        """Store latest scene description"""
        self.latest_description = msg.data
        self.get_logger().info(f'📝 New description: {msg.data}')
    
    def get_latest_frame(self):
        """Get latest frame as JPEG bytes"""
        with self.frame_lock:
            if self.latest_frame is None:
                # Create a simple placeholder image
                placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
                placeholder[:] = (50, 50, 50)  # Dark gray
                
                # Add text
                cv2.putText(placeholder, 'Waiting for camera...', (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
                
                _, buffer = cv2.imencode('.jpg', placeholder)
                return buffer.tobytes()
            
            _, buffer = cv2.imencode('.jpg', self.latest_frame)
            return buffer.tobytes()
    
    def send_goal(self, goal_text: str):
        """Publish goal to VLM planner"""
        msg = String()
        msg.data = goal_text
        self.goal_pub.publish(msg)
        self.get_logger().info(f'🎯 Sent goal: {goal_text}')


# Global node instance (needed for Flask callbacks)
node_instance = None


# Find template directory
def get_template_dir():
    """Find the templates directory in various possible locations"""
    
    # Try installed location first
    possible_paths = [
        # Installed package location
        os.path.join(os.path.dirname(__file__), 'templates'),
        os.path.join(os.path.dirname(__file__), '..', 'templates'),
        os.path.join(os.path.dirname(__file__), '..', '..', 'templates'),
        # Source location
        '/home/teja/spiderbot_ws/src/spiderbot_vlm/templates',
        # Installed location
        '/home/teja/spiderbot_ws/install/spiderbot_vlm/lib/spiderbot_vlm/templates',
        '/home/teja/spiderbot_ws/install/spiderbot_vlm/share/spiderbot_vlm/templates',
    ]
    
    for path in possible_paths:
        full_path = os.path.abspath(path)
        dashboard_html = os.path.join(full_path, 'Dashboard.html')
        if os.path.isfile(dashboard_html):
            print(f"✓ Found templates at: {full_path}")
            return full_path
    
    # If not found, print error and use current directory
    print("⚠ WARNING: Could not find templates directory!")
    print("Searched in:")
    for path in possible_paths:
        print(f"  - {os.path.abspath(path)}")
    print("\nUsing current directory as fallback.")
    return os.path.dirname(__file__)


# Initialize Flask with proper template directory
template_dir = get_template_dir()
app = Flask(__name__, template_folder=template_dir)
CORS(app)


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('Dashboard.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    def generate():
        while True:
            if node_instance is not None:
                frame = node_instance.get_latest_frame()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)  # ~30 FPS
    
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/get_description')
def get_description():
    """Get latest scene description"""
    if node_instance is not None:
        return jsonify({
            'description': node_instance.latest_description,
            'timestamp': time.time()
        })
    return jsonify({'description': 'Node not ready', 'timestamp': 0})


@app.route('/send_command', methods=['POST'])
def send_command():
    """Send command to robot"""
    data = request.json
    command = data.get('command', '').strip()
    
    if not command:
        return jsonify({'success': False, 'error': 'Empty command'})
    
    if node_instance is not None:
        node_instance.send_goal(command)
        return jsonify({'success': True, 'command': command})
    
    return jsonify({'success': False, 'error': 'Node not ready'})


@app.route('/describe_scene', methods=['POST'])
def describe_scene():
    """Request scene description"""
    if node_instance is not None:
        node_instance.send_goal("describe the scene")
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Node not ready'})


def run_flask():
    """Run Flask app in separate thread"""
    app.run(host='0.0.0.0', port=node_instance.port, debug=False, threaded=True)


def main():
    global node_instance
    
    rclpy.init()
    node_instance = VLMDashboardNode()
    
    # Start Flask in background thread
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    
    node_instance.get_logger().info('🚀 Dashboard server started!')
    node_instance.get_logger().info(f'   Open: http://localhost:{node_instance.port}')
    
    try:
        rclpy.spin(node_instance)
    except KeyboardInterrupt:
        node_instance.get_logger().info('Shutting down dashboard...')
    finally:
        node_instance.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()