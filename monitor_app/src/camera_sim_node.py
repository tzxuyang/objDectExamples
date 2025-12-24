#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time
import logging

logging.basicConfig(level=logging.INFO)

_VIDEO_PATH = '/home/yang/MyRepos/object_detection/videos/port_0002.mp4'

class ImagePublisher(Node):
    def __init__(self, img_size, video_path):
        super().__init__('image_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/camera/color/image_rect_raw', 25)
        self.video_path = video_path # Replace with your video file
        width, height = img_size
        self.width = width
        self.height = height
        self.cv_image = None
        self.bridge = CvBridge()

    def get_image(self, cap):
        ret, frame = cap.read()
        logging.info(f"Total frames in video: {ret}")
        if frame is not None:
            self.cv_image = cv2.resize(frame, (self.width, self.height))
    
    def pulish_msg(self):
        self.publisher_.publish(self.bridge.cv2_to_imgmsg(np.array(self.cv_image), "bgr8"))
        self.get_logger().info('Publishing an image')

    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        while rclpy.ok():
            self.get_image(cap)
            cv2.imshow("Camera", self.cv_image)
            cv2.waitKey(1)
            self.pulish_msg()
            time.sleep(0.0313)
    
def main(args=None):
    rclpy.init(args=args)
    image_publisher = ImagePublisher(img_size = (424, 240), video_path = _VIDEO_PATH)
    logging.info("Starting image publisher...")
    try:
        image_publisher.run()
    except KeyboardInterrupt:
        pass
    finally:
        image_publisher.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
   main()