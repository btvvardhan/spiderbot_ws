#!/usr/bin/env python3
"""
ROS2 Bag to MP4 Converter

Converts a ROS2 bag file containing image messages to MP4 video

Usage:
  python3 bag_to_mp4.py /path/to/bag/folder output.mp4
  
  # Or just the bag folder (auto-generates output name):
  python3 bag_to_mp4.py /path/to/bag/folder
"""

import sys
import os
import rclpy
from rclpy.node import Node
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


class BagToVideoConverter:
    def __init__(self, bag_path, output_file, topic='/camera/image_raw'):
        self.bag_path = bag_path
        self.output_file = output_file
        self.topic = topic
        self.bridge = CvBridge()
        self.writer = None
        self.frame_count = 0
        
    def convert(self):
        """Convert bag to video"""
        print(f'🎬 ROS2 Bag to MP4 Converter')
        print(f'   📂 Input:  {self.bag_path}')
        print(f'   💾 Output: {self.output_file}')
        print(f'   📷 Topic:  {self.topic}')
        print()
        
        # Open bag
        storage_options = rosbag2_py.StorageOptions(
            uri=self.bag_path,
            storage_id='sqlite3'
        )
        
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr'
        )
        
        reader = rosbag2_py.SequentialReader()
        reader.open(storage_options, converter_options)
        
        # Get topic types
        topic_types = reader.get_all_topics_and_types()
        type_map = {topic.name: topic.type for topic in topic_types}
        
        if self.topic not in type_map:
            print(f'❌ Error: Topic {self.topic} not found in bag!')
            print(f'Available topics: {list(type_map.keys())}')
            return False
        
        print(f'✓ Found topic: {self.topic}')
        print(f'✓ Processing frames...')
        print()
        
        # Read messages
        while reader.has_next():
            (topic, data, timestamp) = reader.read_next()
            
            if topic == self.topic:
                # Deserialize message
                msg_type = get_message(type_map[topic])
                msg = deserialize_message(data, msg_type)
                
                # Convert to OpenCV image
                try:
                    cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                    
                    # Initialize writer on first frame
                    if self.writer is None:
                        height, width = cv_image.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.writer = cv2.VideoWriter(
                            self.output_file,
                            fourcc,
                            30.0,  # FPS
                            (width, height)
                        )
                        print(f'   📐 Resolution: {width}x{height}')
                    
                    # Write frame
                    self.writer.write(cv_image)
                    self.frame_count += 1
                    
                    # Progress
                    if self.frame_count % 30 == 0:
                        duration = self.frame_count / 30.0
                        print(f'   ⏱️  {self.frame_count:5d} frames ({duration:6.1f}s)', end='\r')
                
                except Exception as e:
                    print(f'\n⚠ Warning: Failed to convert frame: {e}')
        
        # Cleanup
        if self.writer is not None:
            self.writer.release()
        
        # Stats
        duration = self.frame_count / 30.0
        
        try:
            filesize = os.path.getsize(self.output_file) / (1024 * 1024)
            size_str = f'{filesize:.1f} MB'
        except:
            size_str = 'unknown'
        
        print()
        print()
        print('✅ Conversion complete!')
        print(f'   📊 Frames:   {self.frame_count}')
        print(f'   ⏱️  Duration: {duration:.1f}s')
        print(f'   💾 Size:     {size_str}')
        print(f'   📁 Saved to: {self.output_file}')
        
        return True


def main():
    if len(sys.argv) < 2:
        print('Usage: python3 bag_to_mp4.py <bag_folder> [output.mp4]')
        print()
        print('Example:')
        print('  python3 bag_to_mp4.py ~/spiderbot_ws/rosbag2_2025_12_01-18_21_18')
        print('  python3 bag_to_mp4.py ~/spiderbot_ws/rosbag2_2025_12_01-18_21_18 my_video.mp4')
        sys.exit(1)
    
    bag_path = sys.argv[1]
    
    # Generate output filename if not provided
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    else:
        # Extract date from bag folder name
        bag_name = os.path.basename(bag_path.rstrip('/'))
        output_file = f'{bag_name}.mp4'
    
    # Check bag exists
    if not os.path.exists(bag_path):
        print(f'❌ Error: Bag folder not found: {bag_path}')
        sys.exit(1)
    
    # Convert
    converter = BagToVideoConverter(bag_path, output_file)
    success = converter.convert()
    
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
