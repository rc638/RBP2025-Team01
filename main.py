#!/usr/bin/env python3
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError

import cv2
import numpy as np

class DetermineColor(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.image_sub = self.create_subscription(Image, '/color', self.callback, 10)
        self.color_pub = self.create_publisher(Header, '/rotate_cmd', 10)
        self.bridge = CvBridge()

    def callback(self, data):
        self.get_logger().info("Image received")
        msg = Header()
        msg = data.header
        msg.frame_id = '0'  # 기본값: STOP

        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 밝은 영역만 추출 (모니터)
            _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # 가장 큰 밝은 영역 선택
                monitor_contour = max(contours, key=cv2.contourArea)
                rect = cv2.minAreaRect(monitor_contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)

                # Perspective transform
                box = self.order_points(box)
                dst_size = 200
                dst_pts = np.array([
                    [0, 0],
                    [dst_size - 1, 0],
                    [dst_size - 1, dst_size - 1],
                    [0, dst_size - 1]
                ], dtype='float32')
                M = cv2.getPerspectiveTransform(box, dst_pts)
                warped = cv2.warpPerspective(image, M, (dst_size, dst_size))
                hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

                # 색상 범위 정의
                lower_blue = np.array([85, 50, 30])
                upper_blue = np.array([140, 255, 255])
                lower_red1 = np.array([0, 50, 30])
                upper_red1 = np.array([10, 255, 255])
                lower_red2 = np.array([160, 50, 30])
                upper_red2 = np.array([180, 255, 255])

                mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
                mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
                mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
                mask_red = cv2.bitwise_or(mask_red1, mask_red2)

                total_pixels = warped.shape[0] * warped.shape[1]
                blue_count = cv2.countNonZero(mask_blue)
                red_count = cv2.countNonZero(mask_red)
                other_count = total_pixels - (blue_count + red_count)

                self.get_logger().info(f"blue={blue_count}, red={red_count}, other={other_count}")

                if blue_count > red_count and blue_count > other_count:
                    msg.frame_id = '+1'
                elif red_count > blue_count and red_count > other_count:
                    msg.frame_id = '-1'
                else:
                    msg.frame_id = '0'

            else:
                self.get_logger().info("No bright region found.")

        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {e}')
        except Exception as e:
            self.get_logger().error(f'Unhandled exception: {e}')

        # 항상 publish!
        self.color_pub.publish(msg)
        self.get_logger().info(f'Published rotate_cmd: {msg.frame_id}')

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype='float32')
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

def main(args=None):
    rclpy.init(args=args)
    node = DetermineColor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
