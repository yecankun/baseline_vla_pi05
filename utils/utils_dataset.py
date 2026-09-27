import os
import json

import cv2 as cv
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class RawDataset(Dataset):
    
    def __init__(self, exp_folder, is_reinforce=False):
        
        super().__init__()
        self.step_folder_list = os.listdir(exp_folder)
        if ".DS_Store" in self.step_folder_list:
            self.step_folder_list.remove(".DS_Store")
        self.step_folder_list.sort(key=lambda x:int(x[6:10]))
        if is_reinforce:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(480),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                transforms.RandomResizedCrop(size=224, scale=(0.8, 1), ratio=(1, 1)),
                transforms.RandomRotation(degrees=10, center=(112, 112)),
                transforms.ToTensor(),
            ])
        else:
            self.tf = transforms.Compose([
                transforms.ToPILImage(),
                transforms.CenterCrop(480),
                transforms.Resize(224),
                transforms.ToTensor(),
            ])
        self.exp_folder = exp_folder
        
    def __len__(self):
        
        return len(self.step_folder_list)
    
    def __getitem__(self, index):
        
        step_folder = os.path.join(self.exp_folder, self.step_folder_list[index])
        img = cv.imread(os.path.join(step_folder, "master_img.png"))
        img = self.tf(img)
        with open(os.path.join(step_folder, "data.json")) as f:
            data = json.load(f)
        deg = torch.tensor(data["master_deg"])
            
        return img, deg
    

if __name__ == "__main__":
    
    ds = RawDataset("/home/dengxutian/Code/2024_haptic_digital_twin/tmp/dataset/guidewire", False)
    ds_loader = DataLoader(ds, batch_size=1, num_workers=1, shuffle=False)
    
    for img, deg in ds_loader:
        
        img_np = img.detach().cpu().numpy()
        img_now = img_np[0]
        img_now = np.transpose(img_now, (1,2,0))
        print(img_now.shape)

        cv.imshow("img_now", img_now)
        cv.waitKey(0)
    