#!/usr/bin/env python
import torch
from torch.autograd import Variable
import torch.nn as nn
# import torch.nn.functional as F
# import torchvision.models as models

# from wildcat.olive import OliveCounting
# from wildcat.apple import AppleCounting
# from wildcat.almond import AlmondCounting
# from wildcat.car import CarCounting
# from wildcat.person import PersonCounting

import csv
import cv2
import os
import os.path

import torchvision.transforms as transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
# from torch.nn.modules.batchnorm import BatchNorm1d
#from util import conditioned_rmse, interval_rmse, init_dataset, set_seeds
from roma_confident_scount_ros.util import conditioned_rmse, interval_rmse, init_dataset, set_seeds
from roma_confident_scount_ros.engines.base_engine import base_engine
from roma_confident_scount_ros.dataset.fruit_count_dataset import FruitCounting
from PIL import Image

#countingClasses = 4
#hotEncoded = True


class SCOUNT_Engine(base_engine):

    def __init__(self, model, train_set=None, validation_set=None, test_set=None, seed=123, batch_size=4,
                 workers=4, on_GPU=False, save_path='', log_path='', lr=0.0001, lrp=0.1,
                 momentum=0.9, weight_decay=1e-4, num_epochs=50, countClasses = 4, hotEncoded = True):
        super(SCOUNT_Engine, self).__init__()

        self.model = model
        self.train_set = train_set
        self.validation_set = validation_set
        self.test_set = test_set
        self.seed = seed
        self.batch_size = batch_size
        self.workers = workers
        self.on_GPU = on_GPU
        self.save_path = save_path
        self.log_path = log_path
        self.lr = lr
        self.lrp = lrp
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.countClasses = countClasses
        self.hotEncoded = hotEncoded

        self.transform = None

        self.useMultiGPU = True
        self.device_ids = [0, 1, 2, 3]

        print("Count classes ", countClasses)

        self.validation_error_diagonal = -1

    def train_net(self
                  # train_set_,
                  #       val_set_,
                  #       writer,
                  #       seed_,
                  #       save_path
                  ):

        writer = SummaryWriter(self.log_path)
        best_validation_error = 1000.0
        best_validation_error_diagonal = 1000.0
        set_seeds(self.seed)

        # dataset loaders
        train_dataset, train_loader = init_dataset(self.train_set, train=True, batch_size=self.batch_size, workers=self.workers)
        test_dataset, test_loader = init_dataset(self.validation_set, train=False, batch_size=self.batch_size, workers=self.workers)

        # load model
        self.model = self.model.train()
        if self.on_GPU:
            if self.useMultiGPU:
                model = torch.nn.DataParallel(self.model, device_ids=self.device_ids).cuda()
            else:
                self.model = self.model.cuda()


        # define optimizer
        optimizer = torch.optim.SGD(self.model.get_config_optim(self.lr, self.lrp),
                                    lr=self.lr,
                                    momentum=self.momentum,
                                    weight_decay=self.weight_decay)

        criterion = nn.MSELoss()
        j = 0



        for epoch in range(self.num_epochs):  # loop over the dataset multiple times

            epoc_num = epoch+1
        
            #if(epoc_num < 48):
                #continue
        
            print('epoch %d of %d' % (epoc_num, self.num_epochs))


            train_loader = tqdm(train_loader, desc='Training')
            self.model.train()

            for i, data in enumerate(train_loader):
                # get the inputs
                inputs_datas, labels = data
                inputs, img_names = inputs_datas

                # wrap them in Variable
                if self.on_GPU:
                    inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.model.forward(inputs)
                '''
                print('outputs')
                print(outputs)
                print('labels')
                print(labels)
                '''
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                n_iter = j
                writer.add_scalar('Train/Loss', loss.data, n_iter)
                j += 1

            # get validation error
            with torch.no_grad():
                validation_error = self.validate_current_model(val_loader=test_loader)




            print("validation error: ", validation_error)
            print("diagonal validation error: ", self.validation_error_diagonal)

            # Best error
            if validation_error < best_validation_error:
                best_validation_error = validation_error

                # save a new model for each epoch
                #print('saving model: %s_epoch_%d' % (self.save_path, epoch))
                print('saving best model: %s/seed_%d_best_checkpoint_epoch_%d.pth' % (self.save_path, self.seed, epoc_num))
                #torch.save(self.model.state_dict(), ('%s/seed_%d_best_checkpoint.pth' % (self.save_path, self.seed)))
                torch.save(self.model.state_dict(), ('%s/seed_%d_best_checkpoint_epoch_%d.pth' % (self.save_path, self.seed, epoc_num)))


            # Best diagonal error
            if self.validation_error_diagonal >= 0 and self.validation_error_diagonal < best_validation_error_diagonal:
                best_validation_error_diagonal = self.validation_error_diagonal
                print('saving best diagonal model: %s/seed_%d_diagonal_checkpoint_epoch_%d.pth' % (self.save_path, self.seed, epoc_num))
                torch.save(self.model.state_dict(), ('%s/seed_%d_diagonal_checkpoint_epoch_%d.pth' % (self.save_path, self.seed, epoc_num)))

            # save every 50 epochs and the last one
            if epoc_num % 50 == 0:
                print('saving 50 model: %s/seed_%d_last_checkpoint_epoch_%d.pth' % (self.save_path, self.seed, epoc_num))
                torch.save(self.model.state_dict(), ('%s/seed_%d_50_checkpoint_epoch_%d.pth' % (self.save_path, self.seed, epoc_num)))

            if epoc_num == self.num_epochs:
                print('saving last model: %s/seed_%d_last_checkpoint_epoch_%d.pth' % (self.save_path, self.seed, epoc_num))
                torch.save(self.model.state_dict(), ('%s/seed_%d_last_checkpoint_epoch_%d.pth' % (self.save_path, self.seed, epoc_num)))

        print('Finished Training')
        return self.model, best_validation_error


    def loadNetwork(self, path):
        #torch.save(self.model.state_dict(), ('%s/seed_%d_best_checkpoint' % (self.save_path, self.seed)))
        print("Loading network")
        #self.model.load_state_dict(torch.load(path))
        self.model.load_state_dict(torch.load(path, map_location='cpu'))#, pickle_module=pickle, **pickle_load_args))
        #self.model.load_state_dict(torch.load('/home/mrs/git/WS-COUNT/models/seed_1_best_checkpoint.pth'))
        #print(torch.load('/home/cscarbone/git/WS-COUNT/models/seed_1_best_checkpoint.pth'))
        print("Network loaded")

    def validate_current_model(self, val_loader):
        self.model.eval()
        if self.on_GPU:
            self.model = self.model.cuda()

        val_errors = []
        val_loader = tqdm(val_loader, desc='Testing')
        predictions = []
        labels_list = []

        val_errors_diagonal = []


        for i, data in enumerate(val_loader):
            # get the inputs
            inputs_datas, labels = data
            inputs, img_names = inputs_datas

            # wrap them in Variable
            if self.on_GPU:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward
            outputs = self.model.forward(inputs)

            # convert back to a numpy array
            if self.on_GPU:
                outputs = outputs.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
            else:
                outputs = outputs.data.numpy()
                labels = labels.data.numpy()

            #print("outputs shape " + str(outputs.shape))
            #print("outputs: ")
            #print(outputs)
            '''
            print("outputs")
            print(outputs)
            print(labels)
            '''

            if self.hotEncoded:
                for b in range(outputs.shape[0]):
                    for c in range(outputs.shape[1]):

                        pred = outputs[b,c]
                        lab = float(int(labels[b,c]))


                        if pred < 0.0:
                            pred = 0.0
                        err = lab - pred
                        
                        '''
                        print("lab: ")
                        print(lab)
                        print("pred: ")
                        print(pred)
                        '''

                        val_errors.append(err)
                        predictions.append(pred)
                        labels_list.append(lab)

                        if(lab == 1):
                            val_errors_diagonal.append(err) 


            else:
                for b in range(outputs.shape[0]):

                    lab = round(labels[b,0],10)

                    pred = round(outputs[b, 0], 10)
                    if pred < 0.0:
                        pred = 0.0
                    err = lab - pred
                    
                    '''
                    print("lab: ")
                    print(lab)
                    print("pred: ")
                    print(pred)
                    '''

                    val_errors.append(err)
                    predictions.append(pred)
                    labels_list.append(lab)
        
        # diagonal error
        if self.hotEncoded:
            val_errors_diagonal = np.array(val_errors_diagonal)
            rmse_diagonal = np.sqrt(np.square(val_errors_diagonal).sum()/len(val_errors_diagonal))
            self.validation_error_diagonal = rmse_diagonal

        # best overall error
        val_errors = np.array(val_errors)
        rmse = np.sqrt(np.square(val_errors).sum()/len(val_errors))
        return rmse






    def test_net(self):
        # define dataset
        test_dataset, test_loader = init_dataset(self.test_set, train=False, batch_size=self.batch_size, workers=self.workers)

        self.model.eval()

        if self.on_GPU:
            self.model = self.model.cuda()

        val_errors = []
        val_loader = tqdm(test_loader, desc='Testing')

        val_errors_diagonal = []

        #print("val_loader")
        #print(val_loader)

        confusionMatrix = np.zeros((self.countClasses,self.countClasses), dtype=float)
        confusionClassification =  np.zeros((self.countClasses,self.countClasses), dtype=float)
        confusionCount =  np.zeros((self.countClasses), dtype=float)
        
        predictions = []
        labels_list = []
        for i, data in enumerate(val_loader):

            # get the inputs
            inputs_datas, labels = data
            inputs, img_names = inputs_datas
        
            
            print("datas")
            print(inputs.shape)
            print(torch.tensor(inputs).dtype)
            #print(inputs)
            print(img_names)
            
            # wrap them in Variable
            if self.on_GPU:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            # forward
            outputs = self.model.forward(inputs)
            #print(outputs)
            # convert back to a numpy array
            if self.on_GPU:
                outputs = outputs.data.cpu().numpy()
                labels = labels.data.cpu().numpy()
            else:
                outputs = outputs.data.numpy()
                labels = labels.data.numpy()
                


                
            if self.hotEncoded:
                print("outputs")
                print(outputs.shape)
                print(outputs)
                print(labels.shape)
                print(labels)

                for b in range(outputs.shape[0]):     #batch
                    for c in range(outputs.shape[1]): #class

                        pred = outputs[b,c]
                        lab = float(int(labels[b,c]))
                        
                        #print(pred)
                        #print(lab)

                        if pred < 0.0:
                            pred = 0.0
                        err = lab - pred

                        val_errors.append(err)
                        predictions.append(pred)
                        labels_list.append(lab)

                        if lab == 1:
                            #print("confusion matrix")
                            confusionCount[c] += 1
                            #print(np.amax(outputs[b]))

                            for o in range(outputs.shape[1]):
                                confusionMatrix[o][c] += outputs[b][o]
                                #print(outputs[b][o])

                            val_errors_diagonal.append(err) 

                            classPredicted = np.where(outputs[b] == np.amax(outputs[b]))[0][0]
                            #print(classPredicted)
                            confusionClassification[classPredicted][c] += 1


                            #print(confusionCount)
                            #print(confusionMatrix)
                

                    
                            
            else:
                for b in range(outputs.shape[0]):

                    
                    pred = round(outputs[b,0],10)
                    lab = float(int(labels[b,0]))

                    print(lab)
                    print(pred)
                    
                    
                    if pred < 0.0:
                        pred = 0.0
                    err = lab - pred

                    val_errors.append(err)
                    predictions.append(pred)
                    labels_list.append(lab)

                    #print("pred")
                    #print(pred)
                    
                    

        val_errors = np.array(val_errors)
        predictions = np.array(predictions)
        labels_list = np.array(labels_list)
        
        if self.hotEncoded:
            val_errors_diagonal = np.array(val_errors_diagonal)
            
            for o in range(confusionMatrix.shape[0]):
                for c in range(confusionMatrix.shape[1]):
                    confusionMatrix[o][c] /= confusionCount[c]

        print(val_errors.shape)

        # istogramma delle occorrenze degli errori
        plt.hist(val_errors, bins = 100, color = "skyblue", ec="black")  # arguments are passed to np.histogram
        plt.title("Errors Histogram (error = label - prediction)")
        plt.figure()
        plt.hist(predictions, bins = 100, color = "skyblue", ec="black")  # arguments are passed to np.histogram
        print("%s num samples: %f" % ('test' ,len(val_errors)))

        print("labels sum: ", labels_list.copy().sum())
        print("predictions sum: ", predictions.copy().sum())

        rmse = np.sqrt(np.square(val_errors).sum()/len(val_errors))

        if self.hotEncoded:
            rmse_diagonal = np.sqrt(np.square(val_errors_diagonal).sum()/len(val_errors_diagonal))


        print("root mean squared error: ", rmse)
        print("root mean squared error diagonal: ", rmse_diagonal)

        plt.title(("Predictions Histogram - Root Mean Squared Error = %f" % rmse))
        plt.figure()
        plt.stem(labels_list, markerfmt='bo', label='GT')  # arguments are passed to np.histogram
        plt.stem(predictions, markerfmt='go', label='predictions')  # arguments are passed to np.histogram
        plt.legend()
        plt.title("GT vs Predictions")

        cond_rmse = interval_rmse(predictions, labels_list)

        plt.figure()
        plt.stem(cond_rmse, markerfmt='bo')  # arguments are passed to np.histogram
        plt.title("E2E conditioned rmse")

        #plt.show()


        if self.hotEncoded:

            #plt.figure()

            vmin = 0#min(image.get_array().min() for image in images)
            vmax = 1#max(image.get_array().max() for image in images)
            norm = colors.Normalize(vmin=vmin, vmax=vmax)

            img = plt.matshow(confusionMatrix)
            #plt.title("Detection matrix")
            plt.title("Average probabilities")
            plt.colorbar(orientation='vertical')

            #img.set_norm(norm)

            plt.xlabel('True count')
            plt.ylabel('Detected count')


            for x in range(confusionMatrix.shape[0]):
                for y in range(confusionMatrix.shape[1]):
                    plt.gca().text(y, x, round(confusionMatrix[x][y],5), horizontalalignment='center', verticalalignment='center', fontsize=8);




            img = plt.matshow(confusionClassification)
            #plt.title("Detection matrix")
            plt.title("Total results")
            plt.colorbar(orientation='vertical')

            #img.set_norm(norm)

            plt.xlabel('True count')
            plt.ylabel('Detected count')


            for x in range(confusionClassification.shape[0]):
                for y in range(confusionClassification.shape[1]):
                    plt.gca().text(y, x, int(confusionClassification[x][y]), horizontalalignment='center', verticalalignment='center', fontsize=8);


        # get RSME per class get matrix with absolute values        

        plt.show()



    def doSingleClassification(self, inImage):
        
        print("Classifying")

        self.model.eval()
  

        
        #imgPath = "/home/cscarbone/Dataset/counting_unity/boxes_pov_02/devkit/JPEGImages/1_1_0_1.jpg"
        imgPath = "/home/cscarbone/Dataset/counting_unity/boxes_pov_600/devkit/JPEGImages/99_9_8.jpg"
        img = Image.open(os.path.join(imgPath)).convert('RGB')

        cv_image = cv2.cvtColor(inImage, cv2.COLOR_BGR2RGB)

        cv_image = cv_image[400:800, 600:1000]

        #cv2.imshow("cropped", cv_image)
        #cv2.waitKey(0)

        resize_height = 300
        resize_width = 300

        new_shape = (resize_height,resize_width)
        cv_image = cv2.resize(cv_image, new_shape, interpolation=cv2.INTER_LINEAR)
        
        img = Image.fromarray(cv_image)


        print("image loaded")        

        # image normalization
        image_normalization_mean = [0.485, 0.456, 0.406]
        image_normalization_std = [0.229, 0.224, 0.225]
        # image_normalization_mean = [0.45, 0.45, 0.45]
        # image_normalization_std = [0.22, 0.22, 0.22]
        normalize = transforms.Normalize(mean=image_normalization_mean,
                                         std=image_normalization_std)
            
        myImageTransform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])
       
        tensor_img = myImageTransform(img)


            
        #print('img')
        #print(img.shape)
        #print(img)
            
        
        #convert_tensor = transforms.ToTensor() 
        #tensor_img = convert_tensor(img)
        #tensor_img = tensor_img.int()
        #tensor_img = torch.as_tensor(img, dtype=int)
        #print(torch.Tensor(tensor_img).dtype)
        #print(tensor_img)
        tensor_img = tensor_img.unsqueeze(0)
        output = self.model.forward(tensor_img)
      
        print(output)

        '''        
        # Steps necessary to load image in the same way as the testing function
        inImg = FruitCounting(root='/home/cscarbone/mrs_carbone/src/roma_confident_scount_ros/inData/',
                             set='test')


        test_dataset, test_loader = init_dataset(inImg, train=False, batch_size=1, workers=self.workers)
        val_loader = tqdm(test_loader, desc='Testing')
        for i, data in enumerate(val_loader):

            # get the inputs
            inputs_datas, labels = data
            inputs, img_names = inputs_datas
            
            print("in image datas")
            print(inputs.shape)
            print(torch.tensor(inputs).dtype)
            #print(inputs)
            print(img_names)

            output = self.model.forward(inputs)
            print(output)
        '''

        
        print("img classified")
        #print(output)
        
