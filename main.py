#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header

import cv2
import numpy as np

class DetermineColor(Node):
    def __init__(self):
        super().__init__('color_detector')
        # QoS 설정
        img_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.BEST_EFFORT)
        pub_qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        # 원본/압축 영상 구독
        self.create_subscription(Image,
                                 '/camera/color/image_raw',
                                 self.callback_image,
                                 img_qos)
        self.create_subscription(CompressedImage,
                                 '/camera/color/image_raw/compressed',
                                 self.callback_compressed,
                                 img_qos)
        # rotate_cmd 퍼블리셔
        self.color_pub = self.create_publisher(Header, '/rotate_cmd', pub_qos)
        self.bridge = CvBridge()
        self._last_msg = None
        self.create_timer(0.5, self._timer_publish)

    def callback_image(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Image bridge error: {e}')
            return
        if frame is None or frame.size == 0:
            return
        self._process(frame, msg.header)

    def callback_compressed(self, msg: CompressedImage):
        try:
            frame = self.bridge.compressed_imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(f'Compressed bridge error: {e}')
            return
        if frame is None or frame.size == 0:
            return
        self._process(frame, msg.header)

    def _process(self, frame, header):
        # prepare Header
        msg = Header()
        msg = header
        msg.frame_id = '0'

        # 1) Grayscale + blur + edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        edges = cv2.Canny(blur, 50, 150)

        # 2) Find largest quadrilateral contour
        cnts, _ = cv2.findContours(edges,
                                   cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        best_cnt, best_area = None, 0
        for cnt in cnts:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
            if len(approx) == 4:
                area = cv2.contourArea(approx)
                if area > best_area and area > 15000:
                    best_area = area
                    best_cnt = approx

        # 3) Compute ROI: masked monitor region or full frame
        if best_cnt is not None:
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [best_cnt], -1, 255, -1)
            roi = cv2.bitwise_and(frame, frame, mask=mask)
        else:
            roi = frame

        # 4) Crop center 80% to avoid edges
        H, W = roi.shape[:2]
        m = 0.1
        y1, y2 = int(H * m), int(H * (1 - m))
        x1, x2 = int(W * m), int(W * (1 - m))
        crop = roi[y1:y2, x1:x2]

        # 5) Median blur + convert to HSV
        blur2 = cv2.medianBlur(crop, 5)
        hsv = cv2.cvtColor(blur2, cv2.COLOR_BGR2HSV)

        # 6) Build valid-pixel mask: sufficient saturation & brightness
        s = hsv[:,:,1]
        v = hsv[:,:,2]
        valid_mask = cv2.bitwise_and(
            cv2.inRange(s, 50, 255),
            cv2.inRange(v, 50, 255)
        )

        # 7) Hue histogram on valid pixels
        hist = cv2.calcHist([hsv], [0], valid_mask, [180], [0,180]).flatten()
        total = hist.sum()
        if total > 0:
            dh = int(np.argmax(hist))
            ratio = hist[dh] / total
            # 8) Decision thresholds
            if 100 <= dh <= 130 and ratio > 0.05:
                msg.frame_id = '+1'
            elif ((dh <= 5 or dh >= 175) and ratio > 0.05):
                msg.frame_id = '-1'
            else:
                msg.frame_id = '0'

        # 9) Publish & print
        self.color_pub.publish(msg)
        print(f"color_detector: color: {msg.frame_id}", flush=True)
        self._last_msg = msg

    def _timer_publish(self):
        if self._last_msg is not None:
            self.color_pub.publish(self._last_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DetermineColor()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
