'''
from dataset.fruit_count_dataset import FruitCounting
from models.SCOUNT import SCOUNT
from engines.SCOUNT_Engine import SCOUNT_Engine
from configs import configs
'''
import torch

if __name__ == "__main__":
    torch.set_printoptions(edgeitems=800)
    
    #torch.cuda.empty_cache()
    #torch.cuda.set_per_process_memory_fraction(0.9, 0)


    print("horray")
