
from dataset.fruit_count_dataset import FruitCounting
from models.SCOUNT import SCOUNT
from engines.SCOUNT_Engine import SCOUNT_Engine
from configs import configs
import torch

if __name__ == "__main__":
    torch.set_printoptions(edgeitems=800)
    
    torch.cuda.empty_cache()
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
                   subsampled_dim2=conf.subsampled_dim2)
    engine = SCOUNT_Engine(model=model, train_set=train_set, validation_set=test_set, test_set=test_set, seed=1,
                           batch_size=1, save_path=save_path, log_path=log_path, num_epochs=conf.epochs, countClasses = conf.countClasses, hotEncoded = conf.hotEncoded)
    #results = engine.train_net()
    #engine.loadNetwork('/home/cscarbone/git/roma_confident_scount/models/seed_1_best_checkpoint.pth')
    engine.loadNetwork('/home/cscarbone/mrs_carbone/src/roma_confident_scount_ros/test.pth')
    engine.test_net()
