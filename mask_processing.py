import os
import json
import yaml
import numpy as np
import cv2 as cv
import shutil
from PIL import Image


class Masker():
    def __init__(self, train_sample_ratio = 0.8):
        with open('config.yml','r') as cf:
            self.cfg = yaml.load(cf, Loader=yaml.Loader)
        
        self.path = self.cfg['parent_data_dir']
        self.path = os.path.join(self.path, 'dataset')
        os.mkdir(self.path)
        self.binary_mask_path = os.path.join(self.path, 'binary_masks')
        self.instance_seg_path = os.path.join(self.path, 'instance_segmentation')
        self.images_path = os.path.join(self.path, 'images')
        
        self.binary_mask()
        self.train_val_split(train_sample_ratio)

    def binary_mask(self):
        os.mkdir(self.binary_mask_path)
        os.mkdir(self.images_path)
        os.mkdir(self.instance_seg_path)
        counter = 0
        for json_file in self.cfg["labels_path"]:     
            print(f'Pre-processing of {json_file}')
            with open(json_file) as file:
                for line in file:
                    json_line = json.loads(line)
                    arr = np.zeros((720,1280), np.uint8)
                    ins_arr = np.zeros((5,720,1280),np.uint8)   # for each lane instacne defines own image channel
                    j = 0
                    for lane in json_line['lanes']:
                        tmp = np.zeros((720,1280), np.uint8)
                        for i in range(0, len(json_line['h_samples'])-1):
                            if lane[i] == -2:
                                continue
                            else:
                                start_point = (lane[i]-1, json_line['h_samples'][i]-1)
                                if lane[i+1] != -2:
                                    end_point = (lane[i+1]-1, json_line['h_samples'][i+1]-1)
                                else:
                                    end_point = (lane[i]-1, json_line['h_samples'][i]-1)
                                cv.line(arr, start_point, end_point, 255, 6)
                                cv.line(tmp, start_point, end_point, 255, 6)
                        ins_arr[j, :, :] = tmp
                        j += 1

                    file_name = "{:04d}.jpg".format(counter)

                    # binary mask save      
                    img = Image.fromarray(arr, 'L')
                    mask_path = os.path.join(self.binary_mask_path, file_name)
                    img.save(mask_path)

                    # image save
                    image_path = os.path.join(self.images_path, file_name)
                    shutil.copy(self.cfg['parent_data_dir'] + 'data/train_set/' + json_line['raw_file'], image_path)

                    # instance mask save       
                    ins_mask_path = os.path.join(self.instance_seg_path, "{:04d}.npy".format(counter))
                    np.save(ins_mask_path, ins_arr)                    

                    counter += 1
        print("Pre-processing Successfully Completed!")

    def train_val_split(self, train_sample_ratio):
        train_imgs_path = os.path.join(self.path, "train_img")
        train_masks_path = os.path.join(self.path, "train_masks")
        val_imgs_path = os.path.join(self.path, "val_imgs")
        val_masks_path = os.path.join(self.path, "val_masks")
        os.mkdir(train_imgs_path)
        os.mkdir(train_masks_path)
        os.mkdir(val_imgs_path)
        os.mkdir(val_masks_path)
        
        # get all image files for each class
        imagesdir = os.listdir(self.images_path)
        masksdir = os.listdir(self.binary_mask_path)
        
        # calculate the number of images to use for training and validation for each class
        images_train_count = int(len(imagesdir) * train_sample_ratio)
        masks_train_count = int(len(masksdir) * train_sample_ratio)
        
        # class_1_val_count = len(imagesdir) - images_train_count
        # class_2_val_count = len(masksdir) - masks_train_count
        
        # copy the images into the train and validation directories for each class
        for i, file_name in enumerate(imagesdir):
            if i < images_train_count:
                shutil.copy(os.path.join(self.images_path, file_name), train_imgs_path)
            else:
                shutil.copy(os.path.join(self.images_path, file_name), val_imgs_path)
                
        for i, file_name in enumerate(masksdir):
            if i < masks_train_count:
                shutil.copy(os.path.join(self.binary_mask_path, file_name), train_masks_path)
            else:
                shutil.copy(os.path.join(self.binary_mask_path, file_name), val_masks_path)

        print("Train and validation split Successfully Completed!")
Masker()