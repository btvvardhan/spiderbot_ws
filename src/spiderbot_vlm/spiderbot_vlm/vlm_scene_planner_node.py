#!/usr/bin/env python3
import json
import time

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge
import cv2

from google import genai
from google.genai import types


class VLMScenePlannerNode(Node):
    """
    Node that:
      - Subscribes to a live camera image
      - Subscribes to a text goal (e.g. 'describe the scene' or
        'identify the horse and move towards it')
      - Uses Gemini VLM to:
          * Describe the scene (natural language)
          * If the instruction implies movement (go/move/approach...),
            output a short sequence of WASDQEX keys
      - Publishes:
          * /scene_description : std_msgs/String
          * /vlm_cmd           : std_msgs/String (single key per msg)
    """



    def __init__(self):
        super().__init__("vlm_scene_planner_node")

        # -------- Parameters --------
        self.declare_parameter("camera_topic", "/camera/image_raw")
        self.declare_parameter("goal_topic", "/vlm_goal")
        self.declare_parameter("description_topic", "/scene_description")
        self.declare_parameter("cmd_topic", "/cmd")
        self.declare_parameter("model_name", "gemini-2.5-flash")
        self.declare_parameter("command_delay", 0.25)  # seconds between keys

        camera_topic = self.get_parameter("camera_topic").value
        goal_topic = self.get_parameter("goal_topic").value
        description_topic = self.get_parameter("description_topic").value
        cmd_topic = self.get_parameter("cmd_topic").value
        self.model_name = self.get_parameter("model_name").value
        self.command_delay = float(self.get_parameter("command_delay").value)

        # -------- Gemini client --------
        try:
            # Uses GEMINI_API_KEY from environment
            self.client = genai.Client()
        except Exception as e:
            self.get_logger().error(f"Failed to init Gemini client: {e}")
            raise

        # -------- ROS I/O --------
        self.bridge = CvBridge()
        self.latest_image_bytes = None

        # Camera subscription
        self.image_sub = self.create_subscription(
            Image,
            camera_topic,
            self.image_callback,
            10,
        )

        # Goal/instruction subscription
        self.goal_sub = self.create_subscription(
            String,
            goal_topic,
            self.goal_callback,
            10,
        )

        # Scene description publisher
        self.description_pub = self.create_publisher(
            String,
            description_topic,
            10,
        )

        # Command publisher (WASDQEX keys)
        self.cmd_pub = self.create_publisher(
            String,
            cmd_topic,
            10,
        )

        self.get_logger().info("🤖 VLMScenePlannerNode ready!")
        self.get_logger().info(f"   📷 camera_topic:      {camera_topic}")
        self.get_logger().info(f"   🎯 goal_topic:        {goal_topic}")
        self.get_logger().info(f"   📝 description_topic: {description_topic}")
        self.get_logger().info(f"   🕹️ cmd_topic:         {cmd_topic}")
        self.get_logger().info(f"   🧠 model_name:        {self.model_name}")

    # ================== Callbacks ==================

    def image_callback(self, msg: Image):
        """Store latest camera frame as JPEG bytes."""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            ok, buf = cv2.imencode(".jpg", cv_img)
            if not ok:
                return
            self.latest_image_bytes = buf.tobytes()
        except Exception as e:
            self.get_logger().warn(f"Failed to convert image: {e}")

    def goal_callback(self, msg: String):
        """
        Called when a new instruction arrives on /vlm_goal.

        Examples:
          - "describe the scene"
          - "identify the horse and move towards it"
          - "go to the table"
        """
        instruction = msg.data.strip()
        if not instruction:
            return

        if self.latest_image_bytes is None:
            self.get_logger().warn("No camera frame yet; cannot call VLM.")
            return

        self.get_logger().info(f"🎯 New VLM goal: '{instruction}'")

        # 1) Describe the scene
        try:
            scene_description = self.describe_scene(self.latest_image_bytes, instruction)
        except Exception as e:
            self.get_logger().error(f"Scene description failed: {e}")
            scene_description = ""

        if scene_description:
            desc_msg = String()
            desc_msg.data = scene_description
            self.description_pub.publish(desc_msg)
            self.get_logger().info(f"📝 Scene: {scene_description}")

        # 2) If instruction implies movement, also plan commands
        if self.instruction_requires_movement(instruction):
            try:
                cmd_list = self.plan_motion(
                    instruction, self.latest_image_bytes, scene_description
                )
            except Exception as e:
                self.get_logger().error(f"Motion planning failed: {e}")
                return

            if not cmd_list:
                self.get_logger().warn("VLM returned empty movement command list.")
                return

            self.send_commands(cmd_list)
        else:
            self.get_logger().info("Instruction appears descriptive only; no motion.")

    # ================== High-level helpers ==================

    def instruction_requires_movement(self, instruction: str) -> bool:
        """
        Simple heuristic: if the text mentions 'go', 'move', 'walk', 'approach',
        treat it as navigation (in addition to describing the scene).
        """
        text = instruction.lower()
        for word in ["go", "move", "walk", "approach", "towards", "toward"]:
            if word in text:
                return True
        return False

    # ================== VLM calls ==================

    def describe_scene(self, image_bytes: bytes, instruction: str) -> str:
        """
        Ask the VLM to describe the scene, with optional bias from the instruction.
        This is pure natural language output.
        """
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        )

        system_prompt = (
            "You are a vision assistant for a quadruped robot.\n"
            "You see the robot's forward-facing camera image.\n"
            "Describe the scene concisely: key objects, their approximate locations "
            "relative to the robot (left/center/right, near/far), and anything that "
            "might matter for navigation.\n"
            "Use at most 3 sentences.\n"
        )

        user_prompt = f"Based on this camera image, describe the scene. Goal: {instruction}"

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                system_prompt,
                image_part,
                user_prompt,
            ],
        )

        return response.text.strip() if response and response.text else ""

    def plan_motion(
        self,
        instruction: str,
        image_bytes: bytes,
        scene_description: str = "",
    ):
        """
        Ask the VLM to produce a JSON list of movement keys (WASDQEX)
        that move the robot toward the requested object (e.g. a horse).
        """
        image_part = types.Part.from_bytes(
            data=image_bytes,
            mime_type="image/jpeg",
        )

        system_prompt = (
            "You are the low-level controller for a small quadruped robot.\n"
            "You see the robot's forward-facing camera image and you know the user's goal.\n"
            "You must decide how the robot should move to achieve that goal.\n\n"
            "The robot is controlled by sending a **sequence of single-character commands**:\n"
            "  'w' = move forward\n"
            "  's' = move backward\n"
            "  'a' = strafe left\n"
            "  'd' = strafe right\n"
            "  'q' = rotate left in place\n"
            "  'e' = rotate right in place\n"
            "  'x' = stop / no-op\n\n"
            "The user may ask, for example: 'identify the horse and move towards it'.\n"
            "First, you must visually locate the requested object in the camera image.\n"
            "If you clearly see the requested object, plan a short sequence of 5–20\n"
            "commands that reasonably moves the robot towards that object.\n"
            "If you DO NOT see the requested object at all, output ['x'] only.\n\n"
            "You MUST output ONLY a JSON array of these single-character commands, for example:\n"
            "  [\"q\", \"q\", \"w\", \"w\", \"w\", \"x\"]\n"
            "No natural language, no explanation, no extra keys – only the JSON array."
        )

        user_prompt = (
            f"User goal: {instruction}\n\n"
            f"Scene description (optional context): {scene_description}\n"
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=[
                system_prompt,
                image_part,
                user_prompt,
            ],
        )

        raw_text = response.text
        self.get_logger().info(f"🧠 Raw movement response: {raw_text}")

        # Parse JSON
        try:
            cmd_list = json.loads(raw_text)
        except json.JSONDecodeError:
            self.get_logger().error("Failed to parse JSON from VLM response.")
            return []

        # Validate commands
        allowed = set(["w", "a", "s", "d", "q", "e", "x"])
        valid_cmds = []
        for item in cmd_list:
            if not isinstance(item, str):
                continue
            c = item.strip().lower()
            if len(c) == 1 and c in allowed:
                valid_cmds.append(c)

        return valid_cmds

    # ================== Command execution ==================

    def send_commands(self, cmd_list):
        """Publish commands one by one (WASDQEX) to cmd_topic."""
        self.get_logger().info(f"🕹️ Executing {len(cmd_list)} commands: {cmd_list}")
        for c in cmd_list:
            msg = String()
            msg.data = c
            self.cmd_pub.publish(msg)
            time.sleep(self.command_delay)


def main():
    rclpy.init()
    node = VLMScenePlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down VLMScenePlannerNode...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
