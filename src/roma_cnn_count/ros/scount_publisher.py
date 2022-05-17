#!/usr/bin/env python
import rospy

from std_msgs.msg import String
from std_msgs.msg import Bool 
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospkg

from roma_quad_ai.msg import Utility

import numpy as np

class ScountPublisher():
    def __init__(self, inPublish_classified="None"):
        self.publishTopic_classified_=inPublish_classified
        self.image_classified_ = Utility()

        print("Setting cnn classification publish in topic " + self.publishTopic_classified_)
        self.pub_ = rospy.Publisher(self.publishTopic_classified_, Utility, queue_size=10)
   
        self.readyToShare = True

    def publish_classification(self, inImageClassified):
        if self.readyToShare:
            self.image_classified_ = inImageClassified
            self.pub_.publish(self.image_classified_)


