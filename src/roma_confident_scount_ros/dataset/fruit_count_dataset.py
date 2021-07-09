

import csv
#import cv2
import os
import os.path
import tarfile
from urllib.parse import urlparse

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

from roma_confident_scount_ros.configs import configs

#maxCount = 3
#hotEncoded = True
imageFormat = '.jpg'
resize_width = 256
resize_height = 256


def create_object_count_labels_csv(root, dataset, set, file_csv):
    import glob
    
    path_labels = os.path.join(root, 'devkit', dataset, 'csv')
    i = 0

    conf = configs()

    img_list_txt = os.path.join(root, 'devkit', dataset, 'ImageSets', 'Main', '%s.txt' % (set))
    print("creating object count labels csv")    
    print(img_list_txt)
    with open(img_list_txt, 'r') as rf:
        lines = rf.readlines()
    text_file = open(file_csv, "w")
    text_file.write("name,fruits\n")
    for line in lines:  # assuming gif
        i += 1
        filename = path_labels + '/' + line[:-1] + '.csv'
        with open(filename) as f:
            data = f.readlines()
            if not conf.hotEncoded:
                data = data[1:]
            img_name = filename[len(path_labels) + 1:-4]
            print(data)
            if data[0] == '\n':
                count = 0
            else:    
                count = int(len(data))            
            if(count>(conf.countClasses-1) and conf.hotEncoded):
                count = conf.countClasses-1            
            count_label = count
            #print(count_label);
            text_file.write(("%s,%d\n" % (img_name, count_label)))

    text_file.close()







def read_object_labels_csv(file, header=True):
    images = []
    num_categories = 0

    conf = configs()

    print('[dataset] read', file)
    with open(file, 'r') as f:
        reader = csv.reader(f)
        rownum = 0
        for row in reader:
            if header and rownum == 0:
                header = row
            else:
                if num_categories == 0:
                    num_categories = len(row) - 1
                name = row[0]

                #here goes the hot encoded part
                if conf.hotEncoded:
                    labels = np.zeros(conf.countClasses).astype(np.float32)
                    labels[int(row[1:num_categories + 1][0])] = 1
                else:                
                    labels = (np.asarray(row[1:num_categories + 1])).astype(np.float32)
                print(labels)
                #print("labels")
                labels = torch.from_numpy(labels)                
                item = (name, labels)
                #print(item)
                images.append(item)
            rownum += 1
    return images


class FruitCounting(data.Dataset):
    def __init__(self, root, set, transform=None, target_transform=None):
        self.root = root
        self.path_devkit = os.path.join(root, 'devkit')
        self.path_images = os.path.join(root, 'devkit', 'JPEGImages')
        self.set = set
        self.transform = transform
        self.target_transform = target_transform

        #self.maxCountClass = countClasses - 1


        # download dataset
        # download_dataset(self.root)

        # define path of csv file
        path_csv = os.path.join(self.root, 'files')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'counting_' + set + '.csv')
        print(file_csv)
        # create the csv file if necessary
        if not os.path.exists(file_csv):
            print("file_csv not found at: " + file_csv)
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            # labeled_data = read_object_labels(self.root, 'ALMOND', self.set)
            print(file_csv)
            create_object_count_labels_csv(self.root, '', self.set, file_csv)
            # write csv file
            # write_object_labels_csv(file_csv, labeled_data)

        # self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

        print('[dataset] ALMOND counting set=%s number of images=%d' % (
            set,
            len(self.images)))

    def __getitem__(self, index):
        path, target = self.images[index]
        img = Image.open(os.path.join(self.path_images, path + imageFormat)).convert('RGB')
        #new_shape = (resize_height,resize_width)
        #img = cv2.resize(img, new_shape, interpolation=cv2.INTER_LINEAR)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (img, path), target

    def __len__(self):
        return len(self.images)

    # def get_number_classes(self):
    #     return len(self.classes)


class FruitClassificationCnt(data.Dataset):
    def __init__(self, root, set, transform=None, target_class_transform=None, target_count_transform=None):
        self.root = root
        self.path_devkit = os.path.join(self.root, 'devkit')
        self.path_images = os.path.join(self.root, 'devkit', 'JPEGImages')
        self.set = set
        self.transform = transform
        self.target_class_transform = target_class_transform
        self.target_count_transform = target_count_transform

        # download dataset
        # download_dataset(self.root)

        # define path of csv file
        path_csv = os.path.join(self.root, 'files')
        # define filename of csv file
        file_csv = os.path.join(path_csv, 'counting_' + set + '.csv')

        # create the csv file if necessary
        if not os.path.exists(file_csv):
            if not os.path.exists(path_csv):  # create dir if necessary
                os.makedirs(path_csv)
            # generate csv file
            # labeled_data = read_object_labels(self.root, 'ALMOND', self.set)
            create_object_count_labels_csv(self.root, self.set, file_csv)
        # write csv file
        # write_object_labels_csv(file_csv, labeled_data)

        # self.classes = object_categories
        self.images = read_object_labels_csv(file_csv)

        print('[dataset] ALMOND counting set=%s number of images=%d' % (
              set,
              len(self.images)))

    def __getitem__(self, index):
        path, target_count = self.images[index]
        img = Image.open(os.path.join(self.path_images, path + imageFormat)).convert('RGB')

        target_class = torch.FloatTensor([0.0])
        if (torch.autograd.Variable(target_count)).data[0] > 0:
            target_class = torch.FloatTensor([1.0])
        else:
            target_class = torch.FloatTensor([-1.0])

        if self.transform is not None:
            img = self.transform(img)
        if self.target_count_transform is not None:
            target_count = self.target_count_transform(target_count)
        if self.target_class_transform is not None:
            target_class = self.target_class_transform(target_class)

        # return (img, path), target_count, target_class
        return (img, path), target_class

    def __len__(self):
        return len(self.images)

    # def get_number_classes(self):
    #     return len(self.classes)


