
#!/usr/bin/env python
import rospy
from std_msgs.msg import String
import rospkg


#from dataset.fruit_count_dataset import FruitCounting
#from scripts.dataset.fruit_count_dataset import FruitCounting
from roma_confident_scount_ros.dataset.fruit_count_dataset import FruitCounting
from roma_confident_scount_ros.models.SCOUNT import SCOUNT
from roma_confident_scount_ros.engines.SCOUNT_Engine import SCOUNT_Engine
from roma_confident_scount_ros.configs import configs
import torch


def callback(data):
    print("callback loop")
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    
def handle_add_two_ints(req):
    print("Returning [%s + %s = %s]"%(req.a, req.b, (req.a + req.b)))
    return AddTwoIntsResponse(req.a + req.b)
    
def listener():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    
    print("Listener loop start")
    rospy.init_node('counting_network_node', anonymous=True)

    rospy.Subscriber("chatter", String, callback)
    
    print("Subscription done")
    engine.doSingleClassification() 
    
    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()
    
    
    

if __name__ == '__main__':

    print("starting scount node")
    
    torch.set_printoptions(edgeitems=800)
    
    #torch.cuda.empty_cache()
    #torch.cuda.set_per_process_memory_fraction(0.9, 0)

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
                           batch_size=1, save_path=save_path, log_path=log_path, num_epochs=conf.epochs, countClasses = conf.countClasses, hotEncoded = conf.hotEncoded)
    

    rospack = rospkg.RosPack()
    networkPath = (rospack.get_path('roma_confident_scount_ros')) + "/test.pth"

    engine.loadNetwork(networkPath)

    #engine.test_net()
    
    print('node loop starting')
    
    listener()
    
    print("node end")
    
