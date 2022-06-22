#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from std_msgs.msg import Bool 
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import rospkg
#import ros/package.h

from roma_quad_ai.msg import Utility

#from dataset.fruit_count_dataset import FruitCounting
#from scripts.dataset.fruit_count_dataset import FruitCounting
from roma_cnn_count.dataset.fruit_count_dataset import FruitCounting
from roma_cnn_count.models.SCOUNT import SCOUNT
from roma_cnn_count.engines.SCOUNT_Engine import SCOUNT_Engine
from roma_cnn_count.configs import configs

from roma_cnn_count.ros.scount_listener import ScountListener 
from roma_cnn_count.ros.scount_publisher import ScountPublisher 

import torch

import yaml

class nodeLooper():
    def __init__(self):
        self.canClassify = False
        self.classification = []
        self.image_classified = Utility()
        self.uav_count = -2

    def callback(self, data):
        #print("callback loop")
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)

    def callback_image(self, data):
        #print("callback image loop")

        # This is to make the callback only work when the service does a resquest
        if self.canClassify == True:
            self.canClassify = False
            try:
              cv_image = bridge.imgmsg_to_cv2(data, "bgr8")
              self.classification = engine.doSingleClassification(cv_image)
              self.image_classified.detectionCertainties = self.classification
              now = rospy.get_rostime().to_sec()
              self.image_classified.message_Time = now
            except CvBridgeError as e:
              print(e)

         

    def service_image(self, req):
        print("Received request for image classification")
        self.canClassify = True
        
        return TriggerResponse(success=True, message="Request received, classifying now...")

    def handle_add_two_ints(self, req):
        print("Returning [%s + %s = %s]"%(req.a, req.b, (req.a + req.b)))
        return AddTwoIntsResponse(req.a + req.b)
        
    def nodeSetup(self):

        # In ROS, nodes are uniquely named. If two nodes with the same
        # name are launched, the previous one is kicked off. The
        # anonymous=True flag means that rospy will choose a unique
        # name for our 'listener' node so that multiple listeners can
        # run simultaneously.
       
        print("SCount node loop setup start")
        rospy.init_node('counting_network_node', anonymous=True)
        
         
        # spin() simply keeps python from exiting until this node is stopped, used for listener only
        #rospy.spin()
       
     
        print("SCOUNT Node setting listeners")

        #self.uav_count = 1
        scountListeners = []
        scountPublishers = []
        for i in range(self.uav_count):
            # Set listeners
            if i < 9:
                uav_prefix = "/uav_00"
            else:
                uav_prefix = "/uav_0"

            listenTopic_image = uav_prefix + str(i+1) + "/hummingbird/rgb"
            listenService_request = uav_prefix + str(i+1) + "/hummingbird/image_classify"
            newListener = ScountListener(inListenTopic_image = listenTopic_image,
                    inServiceTopic_request = listenService_request )
            scountListeners.append(newListener) 
            #scountListeners[i].request_image()      #FOR TESTING
     
            # Set publishers
            publishTopic_classified = uav_prefix + str(i+1) + "/hummingbird/image_classified"
            newPublisher = ScountPublisher(inPublish_classified = publishTopic_classified)
            scountPublishers.append(newPublisher)

        print('Starting cnn node loop')
        rate = rospy.Rate(1) # 10hz
        while not rospy.is_shutdown():
            #hello_str = "hello world %s" % rospy.get_time()
            #rospy.loginfo(hello_str)

            #print("testing")
            for i in range(self.uav_count):
                inImage_cv = scountListeners[i].get_image()
                if inImage_cv is not None:          
                    print("classifying")
                    self.classification = engine.doSingleClassification(inImage_cv)
                    self.image_classified.detectionCertainties = self.classification
                    now = rospy.get_rostime().to_sec()
                    self.image_classified.message_Time = now
                    
                    scountPublishers[i].publish_classification(inImageClassified = self.image_classified)
                
            rate.sleep() 
        
        

if __name__ == '__main__':

    print("starting scount node")
    bridge = CvBridge()
    
    torch.set_printoptions(edgeitems=800)
    
    #torch.cuda.empty_cache()
    #torch.cuda.set_per_process_memory_fraction(0.9, 0)
    
    #Force cpu even if there is a gpu available
    torch.cuda.is_available = lambda : False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    conf = configs()
    dataset_root = conf.dataset_root
    save_path = conf.SCOUNT_model_path

    train_set = FruitCounting(root=dataset_root,
                              set='train')
    test_set = FruitCounting(root=dataset_root,
                             set='test')
    save_path = save_path
    log_path = save_path + '/log'

    print("training set at: " + dataset_root )
    print("loging at: " + log_path )

    # subsampled_dim1 and subsampled_dim2 are width_img/32 and height_img/32 approximate by excess
    model = SCOUNT(num_classes=1, num_maps=8, subsampled_dim1=conf.subsampled_dim1,
                   subsampled_dim2=conf.subsampled_dim2, countClasses = conf.countClasses, hotEncoded = conf.hotEncoded)
    
    engine = SCOUNT_Engine(model=model, train_set=train_set, validation_set=test_set, test_set=test_set, seed=1,
                    batch_size=1, on_GPU=True, save_path=save_path, log_path=log_path, num_epochs=conf.epochs, countClasses = conf.countClasses, hotEncoded = conf.hotEncoded)
    

    rospack = rospkg.RosPack()
    path_thisPackage = (rospack.get_path('roma_cnn_count'))
    path_network = path_thisPackage + "/test.pth"

    engine.loadNetwork(path_network)
    
    #engine.test_net()
    
    print('creating nodeLooper')

    mainNode = nodeLooper()

    # getting path to main package of simulator
    path_mainPackage = (rospack.get_path('roma_quad_ai'))

    config_path = path_mainPackage + "/swarm_config.yaml"; 
    with open(config_path) as f:
        print(config_path)
        config = yaml.load(f, Loader=yaml.FullLoader)

        mainNode.uav_count = config["num_of_agents"];

    #self.uav_count = 3


    print('node loop starting')

    mainNode.nodeSetup()

    print("node end")
    
