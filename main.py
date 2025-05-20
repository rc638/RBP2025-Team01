#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header

import cv2
import numpy as np

class DetermineColor(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.image_sub = self.create_subscription(Image, '/color', self.callback, 10)
        self.color_pub = self.create_publisher(Header, '/rotate_cmd', 10)
        self.bridge = CvBridge()

    def callback(self, data):
        try:
            # listen image topic
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')

            # prepare rotate_cmd msg
            # DO NOT DELETE THE BELOW THREE LINES!
            msg = Header()
            msg = data.header
            msg.frame_id = '0'  # default: STOP

            # determine background color
            # TODO
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            edges = cv2.Canny(blur, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                self.color_pub.publish(msg)
                return

            max_contour = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(max_contour)
            area = cv2.contourArea(hull)
            if area < 10000:
                self.color_pub.publish(msg)
                return

            mask = np.zeros(gray.shape, dtype=np.uint8)
            cv2.drawContours(mask, [hull], -1, 255, thickness=-1)

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            hsv_masked = cv2.bitwise_and(hsv, hsv, mask=mask)

            lower_blue = np.array([100, 150, 50])
            upper_blue = np.array([140, 255, 255])
            mask_blue = cv2.inRange(hsv_masked, lower_blue, upper_blue)

            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])
            mask_red1 = cv2.inRange(hsv_masked, lower_red1, upper_red1)
            mask_red2 = cv2.inRange(hsv_masked, lower_red2, upper_red2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)

            blue_count = cv2.countNonZero(mask_blue)
            red_count = cv2.countNonZero(mask_red)

            if blue_count > red_count and blue_count > 1000:
                msg.frame_id = '+1'  # CCW
            elif red_count > blue_count and red_count > 1000:
                msg.frame_id = '-1'  # CW
            else:
                msg.frame_id = '0'   # STOP

            # publish color_state
            self.color_pub.publish(msg)
        except CvBridgeError as e:
            self.get_logger().error('Failed to convert image: %s' % e)


if __name__ == '__main__':
    rclpy.init()
    detector = DetermineColor()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()
