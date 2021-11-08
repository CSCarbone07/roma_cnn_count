#!/usr/bin/env python
import rospy

from std_msgs.msg import String
from std_msgs.msg import Bool 
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospkg

import numpy as np

class ScountListener():
    def __init__(self, inListenTopic_image="None", inServiceTopic_request="None"):
        self.listenTopic_image_=inListenTopic_image
        self.listenService_request_=inServiceTopic_request
        self.imageClassified_=None

        self.requestedImage_=False      # image request arrives from agent ai
        self.canUpdateImage_=False      # cnn is ready to update image
        self.imageUpdated_=False        # image is updated
        self.readyToClassify_=True      # this let the node that the listener class has been created


        print("Setting image subscription in topic " + self.listenTopic_image_)
        rospy.Subscriber(self.listenTopic_image_, Image, self.callback_image)
        
        print("Setting cnn image request in service " + self.listenService_request_)
        rospy.Service(self.listenService_request_, Trigger, self.service_image)

        self.bridge = CvBridge()        # cv ros bridge
        self.cv_image_ =  np.zeros((100,100,3), dtype=np.uint8)           # image container to send

        self.DEBUG_ALL = False
        
        print(self.listenTopic_image_)

    def callback_image(self, data):
        if self.DEBUG_ALL: 
            print("callback image loop")

        # This is to make the callback only work when the service does a resquest
        if self.requestedImage_== True and self.imageUpdated_ == False:
            try:
              self.cv_image_ = self.bridge.imgmsg_to_cv2(data, "bgr8")
              self.imageUpdated_=True
              if self.DEBUG_ALL: 
                print("image converted")
            except CvBridgeError as e:
              print(e)

    def service_image(self, req):
        print("Received request for image classification")
        self.canClassify = True
        self.request_image()
        
        return TriggerResponse(success=True, message="Request received, classifying now...")

    # Receive request from other agent to allow callback to update image
    def request_image(self):
        print("image requested")
        self.requestedImage_=True

    # NOT IN USE
    def update_image(self):
        print("updating image for classification")
        self.canUpdateImage_=True
    
    def get_image(self):
        print("getting image")
        if self.cv_image_ is not None and self.imageUpdated_:
            print("returning image")
            #reset coordination variables for new image request
            self.requestedImage_=False
            self.imageUpdated_=False
            return self.cv_image_
        else:
            if self.DEBUG_ALL: 
                print("returning none")
            return None

