
class configs:
    def __init__(self):
        #self.dataset_root = '/home/mrs/Dataset/ISARLab_counting_dataset/almond_ISAR/'
        #self.dataset_root = '/home/mrs/Dataset/counting_wageningen/c_sugarbeets_1n/'
        #self.dataset_root = '/home/mrs/Dataset/counting_wageningen/c_comb_1n/'
        self.dataset_root = '/home/mrs/Dataset/counting_wageningen/c_comb_1n/'
        #self.dataset_root = '/home/mrs/Dataset/counting_wageningen/test_real_sugarbeets/'
        self.SCOUNT_model_path = '/home/mrs/git/WS-COUNT/models'
        self.PAC_model_path = '/home/mrs/git/WS-COUNT/models'
        self.WSCOUNT_model_path = '/home/mrs/git/WS-COUNT/models'

        # subsampled_dim1 and subsampled_dim2 are width_img/32 and height_img/32 approximate by excess
        self.subsampled_dim1 = 10
        self.subsampled_dim2 = 10

        # subsampled_dim1_t4 and subsampled_dim2_t4 are subsampled_dim1/2 and subsampled_dim2/2 approximate by excess
        self.subsampled_dim1_t4 = 5
        self.subsampled_dim2_t4 = 5

        # subsampled_dim1_t16 and subsampled_dim2_t16 are subsampled_dim1_t4/2 and subsampled_dim2_t4/2 approximate by excess
        self.subsampled_dim1_t16 = 3
        self.subsampled_dim2_t16 = 3
