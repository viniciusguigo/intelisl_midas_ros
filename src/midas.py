#!/usr/bin/env python
"""
ROS frontend for Intel-ISL MiDaS adapted from https://github.com/intel-isl/MiDaS
"""
import os
import cv2
import torch
import rospy, rospkg
import numpy as np
import matplotlib.pyplot as plt
from midas.midas_net import MidasNet
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, CompressedImage

class MiDaSROS:
    def __init__(self):
        '''Initialize ros publisher, ros subscriber'''
        # topic where we publish
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/midas/depth/image_raw", Image, queue_size=1)

        # subscribed Topic
        self.subscriber = rospy.Subscriber("/usb_cam/image_raw",
            Image, self.callback, queue_size=1)

        # initialize Intel MiDas
        self.initialized_midas = False
        rospack = rospkg.RosPack()
        ros_pkg_path = rospack.get_path('intelisl_midas_ros')
        model_path = os.path.join(ros_pkg_path, 'src/model-f46da743.pt')

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
        cv2.imshow(window_name, img)
        cv2.waitKey(2)

    def callback(self, img_msg):
        # conversion to OpenCV and the correct color
        img = cv2.cvtColor(
            self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough'), cv2.COLOR_BGR2RGB)
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

            output = prediction.cpu().numpy()
            self.show_image(output, window_name='Estimated Depth')

            # setup message
            msg = Image()
            msg.header.stamp = rospy.Time.now()
            msg.data = output

            # publish
            self.image_pub.publish(msg)


if __name__ == '__main__':
    rospy.init_node('midas_rgb2depth', anonymous=True)
    MiDaSROS()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()