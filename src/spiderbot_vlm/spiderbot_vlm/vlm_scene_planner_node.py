#!/usr/bin/env python3
"""
VLM Scene Planner Node - AI-powered navigation

Receives high-level commands via /vlm_goal
Uses Gemini Vision to:
1. Analyze camera feed
2. Generate movement commands (WASD style)
3. Publish to /cmd for robot execution

Also provides scene descriptions on /scene_description topic
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import google.generativeai as genai
import os
import json
import re
from PIL import Image as PILImage
import io


class VLMScenePlannerNode(Node):
    def __init__(self):
        super().__init__('vlm_scene_planner')
        
        # Parameters
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('goal_topic', '/vlm_goal')
        self.declare_parameter('cmd_topic', '/cmd')
        self.declare_parameter('description_topic', '/scene_description')
        self.declare_parameter('model_name', 'gemini-2.0-flash-exp')
        
        camera_topic = self.get_parameter('camera_topic').value
        goal_topic = self.get_parameter('goal_topic').value
        self.cmd_topic = self.get_parameter('cmd_topic').value
        description_topic = self.get_parameter('description_topic').value
        model_name = self.get_parameter('model_name').value
        
        # Initialize Gemini
        api_key = os.environ.get('GEMINI_API_KEY')
        if not api_key:
            self.get_logger().error('GEMINI_API_KEY not found in environment!')
            self.get_logger().error('Get key from: https://aistudio.google.com/apikey')
            self.get_logger().error('Set with: export GEMINI_API_KEY="your_key_here"')
            raise ValueError('GEMINI_API_KEY required')
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # State
        self.bridge = CvBridge()
        self.latest_image = None
        
        # ROS subscriptions
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10
        )
        
        self.goal_sub = self.create_subscription(
            String,
            goal_topic,
            self.goal_callback,
            10
        )
        
        # ROS publishers
        self.cmd_pub = self.create_publisher(String, self.cmd_topic, 10)
        self.desc_pub = self.create_publisher(String, description_topic, 10)
        
        self.get_logger().info('🤖 VLMScenePlannerNode ready!')
        self.get_logger().info(f'   📷 camera_topic:      {camera_topic}')
        self.get_logger().info(f'   🎯 goal_topic:        {goal_topic}')
        self.get_logger().info(f'   📝 description_topic: {description_topic}')
        self.get_logger().info(f'   🕹️ cmd_topic:         {self.cmd_topic}')
        self.get_logger().info(f'   🧠 model_name:        {model_name}')
    
    def image_callback(self, msg: Image):
        """Store latest camera image"""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        except Exception as e:
            self.get_logger().warn(f'Failed to convert image: {e}')
    
    def goal_callback(self, msg: String):
        """Process new goal from dashboard"""
        goal = msg.data.strip().lower()
        self.get_logger().info(f'🎯 New VLM goal: {goal!r}')
        
        if not self.latest_image:
            self.get_logger().warn('No camera image available yet')
            return
        
        # Check if this is a description request
        if 'describe' in goal:
            self.describe_scene()
        else:
            # Process as movement command
            self.process_movement_goal(goal)
    
    def describe_scene(self):
        """Generate scene description"""
        try:
            # Convert to PIL Image
            pil_image = PILImage.fromarray(self.latest_image)
            
            prompt = """You are a robot with a camera. Describe what you see in this image from your perspective.

Focus on:
- Objects and obstacles in your path
- Available space to move
- Anything blocking your forward movement
- Direction suggestions (left/right/forward)

Keep it concise (2-3 sentences), practical, and from the robot's viewpoint."""
            
            response = self.model.generate_content([prompt, pil_image])
            description = response.text.strip()
            
            self.get_logger().info(f'📝 Scene: {description}')
            
            # Publish description
            msg = String()
            msg.data = description
            self.desc_pub.publish(msg)
            
        except Exception as e:
            self.get_logger().error(f'Failed to describe scene: {e}')
    
    def process_movement_goal(self, goal: str):
        """Convert high-level goal to movement commands"""
        try:
            # Convert to PIL Image
            pil_image = PILImage.fromarray(self.latest_image)
            
            prompt = f"""You are a quadruped spider robot with WASD-style controls. Analyze the camera image and the goal: "{goal}"

Available commands:
- 'w': move forward
- 's': move backward  
- 'a': turn left
- 'd': turn right
- 'q': strafe left
- 'e': strafe right
- 'x': stop

Respond with ONLY a JSON array of commands. Each command is a single character.
Example: ["w", "w", "w", "x"]

For the goal "{goal}", what sequence of commands should you execute?

IMPORTANT: Respond with ONLY valid JSON array, no explanation, no markdown, no code blocks."""
            
            response = self.model.generate_content([prompt, pil_image])
            raw_response = response.text.strip()
            
            self.get_logger().info(f'🧠 Raw movement response: {raw_response}')
            
            # Strip markdown code blocks if present
            commands = self.parse_vlm_response(raw_response)
            
            if not commands:
                self.get_logger().warn('VLM returned empty movement command list.')
                return
            
            self.get_logger().info(f'✅ Parsed commands: {commands}')
            
            # Execute commands
            self.execute_commands(commands)
            
        except Exception as e:
            self.get_logger().error(f'Failed to process movement goal: {e}')
    
    def parse_vlm_response(self, response: str) -> list:
        """Parse VLM response, handling markdown code blocks"""
        try:
            # Remove markdown code blocks (```json ... ```)
            cleaned = re.sub(r'```json\s*', '', response)
            cleaned = re.sub(r'```\s*', '', cleaned)
            cleaned = cleaned.strip()
            
            # Parse JSON
            commands = json.loads(cleaned)
            
            # Validate it's a list
            if not isinstance(commands, list):
                self.get_logger().error(f'Expected list, got {type(commands)}')
                return []
            
            # Validate all commands are single characters
            valid_commands = ['w', 's', 'a', 'd', 'q', 'e', 'x']
            filtered = [cmd for cmd in commands if cmd in valid_commands]
            
            if len(filtered) != len(commands):
                self.get_logger().warn(f'Filtered invalid commands. Original: {commands}, Filtered: {filtered}')
            
            return filtered
            
        except json.JSONDecodeError as e:
            self.get_logger().error(f'Failed to parse JSON from VLM response: {e}')
            self.get_logger().error(f'Response was: {response}')
            return []
        except Exception as e:
            self.get_logger().error(f'Unexpected error parsing VLM response: {e}')
            return []
    
    def execute_commands(self, commands: list):
        """Execute sequence of movement commands"""
        self.get_logger().info(f'🎮 Executing {len(commands)} commands...')
        
        for i, cmd in enumerate(commands):
            msg = String()
            msg.data = cmd
            self.cmd_pub.publish(msg)
            self.get_logger().info(f'  [{i+1}/{len(commands)}] Published: {cmd!r}')
            
            # Small delay between commands
            import time
            time.sleep(0.5)


def main():
    rclpy.init()
    node = VLMScenePlannerNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down VLM planner...')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()