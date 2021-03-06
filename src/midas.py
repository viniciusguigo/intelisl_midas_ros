#!/usr/bin/env python
"""
ROS frontend for Intel-ISL MiDaS adapted from https://github.com/intel-isl/MiDaS
"""
import os, sys
import cv2
import torch
import rospy, rospkg
import numpy as np
import matplotlib.pyplot as plt
from midas.midas_net import MidasNet
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage, CameraInfo

class MiDaSROS:
    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.bridge = CvBridge()
        self.image_depth_pub = rospy.Publisher("/midas/depth/image_raw", Image, queue_size=1)
        self.image_rgb_pub = rospy.Publisher("/midas/rgb/image_raw", Image, queue_size=1)
        self.camera_info_pub = rospy.Publisher("/midas/camera_info", CameraInfo, queue_size=1)

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/midas_rgb/image_raw", Image, self.callback, queue_size=1)

        # setup image display
        self.display_rgb = False
        self.display_depth = True

        # initialize Intel MiDas
        self.initialized_midas = False
        rospack = rospkg.RosPack()
        ros_pkg_path = rospack.get_path('intelisl_midas_ros')
        model_path = os.path.join(ros_pkg_path, 'src/model-f6b98070.pt')

        self.model = MidasNet(model_path, non_negative=True)
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model.to(self.device)
        self.model.eval()
        rospy.loginfo('Loaded Intel MiDaS')

        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self.transform = midas_transforms.default_transform
        rospy.loginfo('Initialized Intel MiDaS transform')
        self.initialized_midas = True

    def show_image(self, img, window_name="Image Window"):
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, img)
        cv2.waitKey(2)

    def callback(self, img_msg):
        # conversion to OpenCV and the correct color
        img = cv2.cvtColor(
            self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough'), cv2.COLOR_BGR2RGB)
        if self.display_rgb:
            self.show_image(img, window_name='Ground Truth RGB')

        # convert RGB to depth using MiDaS
        if self.initialized_midas:
            input_batch = self.transform(img).to(self.device)
            with torch.no_grad():
                prediction = self.model(input_batch)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()
            
            # scale pixel values to display
            omax, omin = prediction.max(), prediction.min()
            prediction = (prediction-omin)/(omax - omin)

            # convert depth prediction to numpy
            output = prediction.cpu().numpy()
            if self.display_depth:
                self.show_image(output, window_name='Estimated Depth')

            # setup message (depth)
            depth_msg = self.bridge.cv2_to_imgmsg(output, encoding="passthrough")

            # setup message camera info
            camera_info_msg = CameraInfo()
            camera_info_msg.header.stamp = img_msg.header.stamp
            camera_info_msg.height = img.shape[0]
            camera_info_msg.width = img.shape[1]

            # publish
            self.image_depth_pub.publish(depth_msg)
            self.image_rgb_pub.publish(img_msg)
            self.camera_info_pub.publish(camera_info_msg)


if __name__ == '__main__':
    rospy.init_node('midas_rgb2depth', anonymous=True)
    MiDaSROS()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()