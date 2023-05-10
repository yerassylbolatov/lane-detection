import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Resize, ToPILImage


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        # self.ins_dir = ins_dir
        self.transform = Compose([
            Resize((256, 512)),
            ToTensor(),
            # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.image_files = sorted(os.listdir(self.image_dir))
        self.mask_files = sorted(os.listdir(self.mask_dir))
        # self.ins_files = sorted(os.listdir(self.ins_dir))
        assert len(self.image_files) == len(self.mask_files)


    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])
        # ins_path = os.path.join(self.ins_dir, self.ins_files[idx])

        # Load image and mask
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        # ins = torch.from_numpy(np.load(ins_path))

        # transform1 = Compose([Resize((256,512))])

        # Apply transformations
        img = self.transform(img)
        mask = self.transform(mask)
        # ins = transform1(ins)

        mask[mask > 0.0] = 1.0 
        return img, mask


# # Define data directories
# image_dir = '/Users/yerassyl/Documents/Thesis/dataset/images'
# mask_dir = '/Users/yerassyl/Documents/Thesis/dataset/binary_masks'
# instance_dir = '/Users/yerassyl/Documents/Thesis/dataset/instance_segmentation'

# # Create custom dataset
# dataset = CustomDataset(image_dir, mask_dir)

# # Create dataloader
# batch_size = 8
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)



# # check whether the dimensions of images and masks were loaded correctly 
# for img, m in dataloader:
#     print(img.shape)
#     print(m.shape)

#     # a = m[0][0][150]
#     # print(a)
#     # print(len(a[a>0.0]))
    
#     # a[a>0.0]=1.0
#     # print(a)
#     # print(len(a[a==1.0]))
#     break


# Tested whether the indexation has a sorted order or not 
# a = dataset.__getitem__(2230)

# t = ToPILImage()
# img1 = t(a[0])
# img1.save('test1_dataloader.jpg')

# img2 = t(a[1])
# img2.save('test2_dataloader.jpg')

# img3 = t(a[2])
# img3.save('test3_dataloader.jpg')


