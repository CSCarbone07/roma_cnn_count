#!/usr/bin/env python
import torch
# from torch.autograd import Variable
import torch.nn as nn
# import torch.nn.functional as F
import torchvision.models as models
#from models.base_model import BaseModel
from roma_confident_scount_ros.models.base_model import BaseModel

#countingClasses = 4
#hotEncoded = True


class SCOUNT(BaseModel):

    def __init__(self, num_classes, num_maps, subsampled_dim1, subsampled_dim2, countClasses = 4, hotEncoded = True):
        super(SCOUNT, self).__init__()
        self.subsampled_dim1 = subsampled_dim1
        self.subsampled_dim2 = subsampled_dim2

        self.countClasses = countClasses
        self.hotEncoded = hotEncoded

        model = models.resnet101(True)

        # features net
        self.features = nn.Sequential(
                                      model.conv1,
                                      model.bn1,
                                      model.relu,
                                      model.maxpool,
                                      model.layer1,
                                      model.layer2,
                                      model.layer3,
                                      model.layer4
                                      )

        # classification layer
        num_features = model.layer4[1].conv1.in_channels
        self.classifier = nn.Sequential(
            nn.Conv2d(num_features, num_classes*num_maps, kernel_size=1, stride=1, padding=0, bias=True)
                      , nn.BatchNorm2d(num_classes*num_maps)
                      , nn.ReLU(inplace=True))

        # counter regressor, composed of fully connected layers
        self.fcs_input = subsampled_dim1*subsampled_dim2*num_maps*num_classes
        self.fcs_output = 1
        #self.countingClasses = 4
        #self.countingClasses = 21
        #self.first_input = 729*775
        #self.first_input = 10*3*300*300
        if self.hotEncoded:
            self.regressor = nn.Sequential(
                                           nn.Linear(self.fcs_input, 1000),
                                           nn.BatchNorm1d(1000),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(1000, 1000),
                                           nn.BatchNorm1d(1000),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(1000, self.countClasses),
                                           nn.Softmax(dim=1)
                                           )
        else:
            self.regressor = nn.Sequential(
                                           nn.Linear(self.fcs_input, 1000),
                                           nn.BatchNorm1d(1000),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(1000, 1000),
                                           nn.BatchNorm1d(1000),
                                           nn.ReLU(inplace=True),
                                           nn.Linear(1000, self.fcs_output)
                                           )


    def forward(self, x):
        '''
        print("first step")
        print(x.shape)
        #print(x)
        '''

        x = self.features(x)
        '''     
        print("features")
        print(x.shape)
        #print(x)
        '''

        x = self.classifier(x)
        '''
        print("classifier")
        print(x.shape)
        #print(x)
        '''

        x = x.view(-1, self.num_flat_features(x))
        '''
        print("viewer")
        print(x.shape)
        print(self.fcs_input)
        print(x)
        '''

        x = self.regressor(x)
        '''
        print("regressor")
        print(x.shape)
        print(x)
        '''

        #print("hot encode")
        #hotEncode=torch.tensor(x)
        #hotEncode[hotEncode<0] = 0
        #print(hotEncode)
        #hotEncode = nn.functional.softmax(hotEncode, dim=0)#, _stacklevel=3, dtype=None)
        #print(hotEncode)
        #print(torch.arange(0, 5) % 3)
        #print(nn.functional.one_hot(x.type(torch.LongTensor), num_classes = 20))

        return x

    # def num_flat_features(self, x):
    #     size = x.size()[1:]  # all dimensions except the batch dimension
    #     num_features = 1
    #     for s in size:
    #         num_features *= s
    #     return num_features

    def get_config_optim(self, lr, lrp):
        return [{'params': self.features.parameters(), 'lr': lr * lrp},
                {'params': self.classifier.parameters(), 'lr': lr},
                {'params': self.regressor.parameters(), 'lr': lr}]
