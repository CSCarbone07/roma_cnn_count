# confidence S-COUNT

Counting network to include confidence of counting class.

Based on code repository for the approch described in "Weakly Supervised Fruit Counting for Yield Estimation Using Spatial Consistency"

https://github.com/isarlab-department-engineering/WS-COUNT

# Installation

OS: Ubuntu 16.04 (other versions are not tested)

Requirements:
1. Python 3.5 or later
2. Pytorch 0.4 or later

# Usage
add parameters values on configs.py file with dataset path, models path ect..

launch script "apps/train_PAC.py" for train PAC Models, "apps/train_SCOUNT.py" for SCOUNT model and "apps/train_WSCOUNT.py".

# Dataset

you can download custom fruit counting datasets at: 

http://sira.diei.unipg.it/supplementary/ws-count/ISARLab_counting_dataset.tar.xz
