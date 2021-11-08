
#!/usr/bin/env python
import rospy    
class configs:
    def __init__(self):
        #self.dataset_root = '/home/mrs/Dataset/ISARLab_counting_dataset/almond_ISAR/'
        #self.dataset_root = '/home/mrs/Dataset/counting_wageningen/c_sugarbeets_1n/'
        #self.dataset_root = '/home/mrs/Dataset/counting_wageningen/c_comb_1n/'
        #self.dataset_root = '/home/mrs/Dataset/counting_wageningen/c_comb_1n/'
        #self.dataset_root = '/home/cscarbone/Dataset/counting_unity/boxes_pov/'
        #self.dataset_root = '/home/cscarbone/Dataset/counting_unity/boxes_pov_600/'
        #self.dataset_root = '/home/cscarbone/Dataset/counting_unity/boxes_pov_02/'
        #self.dataset_root = '/home/mrs/catkin_ws_fm/src/roma_confident_scount_ros/inData/'
        self.dataset_root = '/home/cscarbone/catkin_ws_fm/src/roma_confident_scount_ros/inData/'
        #self.dataset_root = '/home/cscarbone/mrs_carbone/src/roma_confident_scount_ros/inData/'
        #self.dataset_root = '/home/mrs/carlos_workspace/src/roma_confident_scount_ros/inData/'
        #self.dataset_root = '/home/cscarbone/mrs_carbone/src/roma_confident_scount_ros/'
        #self.dataset_root = '/home/cscarbone/Dataset/counting_unity/sugarbeets_potatoes/'
        #self.dataset_root = '/home/mrs/Dataset/counting_unity/sugarbeets_potatoes_alt/'
        #self.dataset_root = '/home/mrs/Dataset/counting_wageningen/test_real_sugarbeets/'
        self.SCOUNT_model_path = '/home/cscarbone/git/roma_confident_scount/models'
        self.PAC_model_path = '/home/cscarbone/git/roma_confident_scount/models'
        self.WSCOUNT_model_path = '/home/cscarbone/git/roma_confident_scount/models'

        # subsampled_dim1 and subsampled_dim2 are width_img/32 and height_img/32 approximate by excess
        self.subsampled_dim1 = 10#10
        self.subsampled_dim2 = 10#10

        # subsampled_dim1_t4 and subsampled_dim2_t4 are subsampled_dim1/2 and subsampled_dim2/2 approximate by excess
        self.subsampled_dim1_t4 = 5#5
        self.subsampled_dim2_t4 = 5#5

        # subsampled_dim1_t16 and subsampled_dim2_t16 are subsampled_dim1_t4/2 and subsampled_dim2_t4/2 approximate by excess
        self.subsampled_dim1_t16 = 3#3
        self.subsampled_dim2_t16 = 3#3


        self.epochs = 100
        self.hotEncoded = True
        self.countClasses = 8
        
        #ROS variables
        self.subscribe_topic = '/pylon_camera_node/image_raw'
        #self.subscribe_topic = 'rgb'
