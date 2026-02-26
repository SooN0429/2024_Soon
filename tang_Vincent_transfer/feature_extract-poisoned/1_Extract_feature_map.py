import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
import data_loader
from config import CFG
# import utils
import numpy as np
import os
import time
from PIL import Image

import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

parser = argparse.ArgumentParser(description="JYT_try_extract_feature_map_latent")

parser.add_argument("--extracted_layer",type=str, default="None",help='Which point of feature map want to extract')
parser.add_argument("--source_fruit",type=str, default="None",help='source fruit')
parser.add_argument("--target_fruit",type=str, default="None",help='target fruit')

opt = parser.parse_args()

# Load the pretrained model
model_ft = models.resnet18(pretrained=True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Resnet18Extractor(nn.Module):
    def __init__(self, extracted_layer):
        super(Resnet18Extractor, self).__init__()

        # Load the pretrained model
        model_resnet18 = models.resnet18(pretrained=True)

        self.extracted_layer = extracted_layer
        self.conv1 = model_resnet18.conv1
        self.bn1 = model_resnet18.bn1
        self.relu = model_resnet18.relu
        self.maxpool = model_resnet18.maxpool
        self.layer1 = model_resnet18.layer1
        self.layer2 = model_resnet18.layer2
        self.layer3 = model_resnet18.layer3
        self.layer4 = model_resnet18.layer4
        self.avgpool = model_resnet18.avgpool

    def forward(self, x):
  
        if self.extracted_layer == "1_point":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
        elif self.extracted_layer == "2_point":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        	
            x = self.layer1[0](x)
            
        elif self.extracted_layer == "3_point":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        	
            x = self.layer1(x)
        	
        elif self.extracted_layer == "4_point":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        	
            x = self.layer1(x)
            x = self.layer2[0](x)
            
        elif self.extracted_layer == "5_point":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        	
            x = self.layer1(x)
            x = self.layer2(x)
            
        elif self.extracted_layer == "6_point":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        	
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3[0](x)
            
        elif self.extracted_layer == "7_point":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            
        elif self.extracted_layer == "8_point":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4[0](x)
            
        elif self.extracted_layer == "9_point":
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
        	
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            
        return x

def load_data(src,tar,root_dir):
    folder_src = root_dir + src
    folder_tar = root_dir + tar
    source_loader = data_loader.load_data(folder_src, CFG['batch_size'], True, CFG['kwargs'])
    target_train_loader = data_loader.load_data(folder_tar, CFG['batch_size'], True, CFG['kwargs'])
    return source_loader,target_train_loader

# for source positive & negative train (all in one data) -> recommend
def source_train_to_feature_map(source_loader):
    Stime = time.time()
    print("Start source_train feature map")
    iter_source = iter(source_loader) 
    len_source_loader = len(source_loader)
    count = 0
    for i in range(len_source_loader): 
            class_source_total = classes_source.data.cpu().numpy() 
        print(i)
        inputs_source, classes_source = iter_source.next()
        inputs_source = inputs_source.to(DEVICE)
        classes_source = classes_source.to(DEVICE)
        if count==0: 
            features_source_total = extractor(inputs_source).data.cpu().numpy() 
            class_source_total = classes_source.data.cpu().numpy() 
            count = 1
        elif count==1: 
            features_source_input = extractor(inputs_source).data.cpu().numpy()  
            classes_source_input = classes_source.data.cpu().numpy() 
            features_source_total = np.vstack((features_source_total,features_source_input))
            class_source_total = np.vstack((class_source_total,classes_source_input))
    Endtime = time.time()
    print('*****Time = ', Endtime-Stime)
    np.save(source_train_save_path + "\\source_train_ColorJitter_feature.npy", features_source_total)
    np.save(source_train_save_path + "\\source_train_ColorJitter_feature_label.npy", class_source_total)

# for target positive train (all in one data) -> recommend
def target_train_to_feature_map(target_loader):
    print("Start target_train feature map")
    iter_target = iter(target_loader)
    len_target_loader = len(target_loader)
    count = 0
    for i in range(len_target_loader):
        print(i)
        inputs_target, classes_target = iter_target.next()
        inputs_target = inputs_target.to(DEVICE)
        classes_target = classes_target.to(DEVICE)
        if classes_target == 1 and count == 0:
            features_target_total = extractor(inputs_target).data.cpu().numpy()
            class_target_total = classes_target.data.cpu().numpy()
            count = 1
        elif classes_target == 1 and count == 1: 
            features_target_input = extractor(inputs_target).data.cpu().numpy()
            classes_target_input = classes_target.data.cpu().numpy()
            features_target_total = np.vstack((features_target_total,features_target_input))
            class_target_total = np.vstack((class_target_total,classes_target_input))
            
    np.save(target_train_save_path + "\\target_train_ColorJitter_feature.npy", features_target_total)
    np.save(target_train_save_path + "\\target_train_ColorJitter_feature_label.npy", class_target_total)

def source_fruit_to_path(source_fruit): 
    source_path_index = "None"
    # print(source_fruit)
    if source_fruit == "apple":
        source_path_index = "apple_to_"

    elif source_fruit == "banana":
        source_path_index = "banana_to_"
    
    elif source_fruit == "carambola":
        source_path_index = "carambola_to_"

    elif source_fruit == "guava":
        source_path_index = "guava_to_"

    elif source_fruit == "muskmelon":
        source_path_index = "muskmelon_to_"

    elif source_fruit == "peach":
        source_path_index = "peach_to_"

    elif source_fruit == "pear":
        source_path_index = "pear_to_" 

    elif source_fruit == "tomato":
        source_path_index = "tomato_to_"

    print(source_path_index)
    return source_path_index

def target_fruit_to_path(target_fruit):
    target_path_index = "None"
    if target_fruit == "apple":
        target_path_index = "to_apple"

    elif target_fruit == "banana":
        target_path_index = "to_banana"
    
    elif target_fruit == "carambola":
        target_path_index = "to_carambola"

    elif target_fruit == "guava":
        target_path_index = "to_guava"

    elif target_fruit == "muskmelon":
        target_path_index = "to_muskmelon"

    elif target_fruit == "peach":
        target_path_index = "to_peach"

    elif target_fruit == "pear":
        target_path_index = "to_pear" 

    elif target_fruit == "tomato":
        target_path_index = "to_tomato"

    print(target_path_index)
    return target_path_index

if __name__ == '__main__': # setup
    root_file_path = "D:\\Tang\\create_datasets\\feature_extract\\dataset\\"

    instance_root_path = root_file_path + "ColorJitter_image\\" 
    instance_source_index = source_fruit_to_path(opt.source_fruit) # 
    instance_target_index = target_fruit_to_path(opt.target_fruit)
    instance_path = instance_root_path + instance_source_index + "\\" + instance_target_index + "\\sn_200_sp_200"
    source_name = "\\source\\train"
    target_name = "\\target\\train"
    print(instance_path) # D:\Tang\testing\feature_extract\dataset\image\apple_to_\to_apple\sn_200_sp_200
    source_train_loader , target_train_loader = load_data(source_name,target_name,instance_path)

    feature_map_path = root_file_path + "feature_map\\latent\\" + opt.extracted_layer[0]
    if not os.path.isdir(feature_map_path):
        os.mkdir(feature_map_path)
    source_train_save_path = feature_map_path + "\\" + instance_source_index + "\\" + instance_target_index + "\\source\\train"
    if not os.path.isdir(source_train_save_path):
        os.makedirs(source_train_save_path)
    target_train_save_path = feature_map_path + "\\" + instance_source_index + "\\" + instance_target_index + "\\target\\train"
    if not os.path.isdir(target_train_save_path):
        os.makedirs(target_train_save_path)

    extracted_layer = opt.extracted_layer
    extractor = Resnet18Extractor(opt.extracted_layer)
    extractor = extractor.to(DEVICE)

    # source_train_to_feature_map(source_train_loader)
    target_train_to_feature_map(target_train_loader)


