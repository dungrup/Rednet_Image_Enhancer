import os
import random
import glob
import io
import numpy as np
from PIL import Image
import torch 

class WaymoDataset(torch.utils.data.Dataset):
    def __init__(self, raw_root, compressed_root, patch_size, train_flag=True):
        self.patch_size = patch_size

        # Find all tfrecord folders
        self.raw_folders = sorted(glob.glob(os.path.join(raw_root, "tfrecord_*", "val")))
        self.compressed_folders = sorted(glob.glob(os.path.join(compressed_root, "tfrecord_*", "val_cbr_5Mbps_5Mbuf", "val")))

        if train_flag:
            self.raw_folders = self.raw_folders[0:50]
            self.compressed_folders = self.compressed_folders[0:50]
        else:
            self.raw_folders = self.raw_folders[50:70]
            self.compressed_folders = self.compressed_folders[50:70]
        
        # Verify matching number of folders
        assert len(self.raw_folders) == len(self.compressed_folders), \
            f"Number of raw folders ({len(self.raw_folders)}) doesn't match compressed folders ({len(self.compressed_folders)})"
        
        # Initialize lists to store all image paths
        self.compressed_images = []
        self.raw_images = []
        
        # Collect all image paths from all folders
        for raw_folder, comp_folder in zip(self.raw_folders, self.compressed_folders):
            # Get images from this folder pair
            compressed_imgs = sorted(glob.glob(os.path.join(comp_folder, "*.png")))
            raw_imgs = sorted(glob.glob(os.path.join(raw_folder, "*.png"))) 
            
            # Verify matching number of images in this folder pair
            assert len(compressed_imgs) == len(raw_imgs), \
                f"Mismatch in images count for folders {raw_folder} ({len(raw_imgs)}) and {comp_folder} ({len(compressed_imgs)})"
            
            self.compressed_images.extend(compressed_imgs)
            self.raw_images.extend(raw_imgs)


    def __getitem__(self, idx):
        # load images and labels
        img_path = self.compressed_images[idx]
        label_path = self.raw_images[idx]
        
        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")
        
        
        image = np.array(img).astype(np.float32)
        image = np.transpose(image, axes=[2, 0, 1])
        image /= 255.0

        target = np.array(label).astype(np.float32)
        target = np.transpose(target, axes=[2, 0, 1])
        target /= 255.0
        
        return image, target

    def __len__(self):
        return len(self.compressed_images)
    

class CommaDataset(torch.utils.data.Dataset):
    def __init__(self, root, patch_size):
        self.patch_size = patch_size
        self.root = root

        # Find all folders
        self.raw_folders = sorted(glob.glob(os.path.join(root, "*", "raw")))
        self.compressed_folders = sorted(glob.glob(os.path.join(root, "*", "h264")))

        # Verify matching number of folders
        assert len(self.raw_folders) == len(self.compressed_folders), \
            f"Number of raw folders {len(self.raw_folders)} doesn't match compressed folders {len(self.compressed_folders)}"

        
        # Initialize lists to store all image paths
        self.compressed_images = []
        self.raw_images = []

        # Collect all image paths from all folders
        for raw_folder, comp_folder in zip(self.raw_folders, self.compressed_folders):
            # Get images from this folder pair
            compressed_imgs = sorted(glob.glob(os.path.join(comp_folder, "*.png")))
            raw_imgs = sorted(glob.glob(os.path.join(raw_folder, "*.png"))) 
            
            # Verify matching number of images in this folder pair
            assert len(compressed_imgs) == len(raw_imgs), \
                f"Mismatch in images count for folders {raw_folder} ({len(raw_imgs)}) and {comp_folder} ({len(compressed_imgs)})"
            
            self.compressed_images.extend(compressed_imgs)
            self.raw_images.extend(raw_imgs)

    def __getitem__(self, idx):
        # load images and labels
        img_path = self.compressed_images[idx]
        label_path = self.raw_images[idx]
        
        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")
        
        
        image = np.array(img).astype(np.float32)
        image = np.transpose(image, axes=[2, 0, 1])
        image /= 255.0

        target = np.array(label).astype(np.float32)
        target = np.transpose(target, axes=[2, 0, 1])
        target /= 255.0
        
        return image, target
    
    def __len__(self):
        return len(self.compressed_images)
    
class BDDDataset(torch.utils.data.Dataset):
    def __init__(self, raw_root, compressed_root, patch_size, train_flag=True):
        self.patch_size = patch_size

        # Find all tfrecord folders
        self.raw_folders = sorted(glob.glob(os.path.join(raw_root, "*")))
        self.compressed_folders = sorted(glob.glob(os.path.join(compressed_root, "*", "val_cbr_1Mbps_1Mbuf", "val")))

        if train_flag:
            self.raw_folders = self.raw_folders[0:50]
            self.compressed_folders = self.compressed_folders[0:50]
        else:
            self.raw_folders = self.raw_folders[50:70]
            self.compressed_folders = self.compressed_folders[50:70]
        
        # Verify matching number of folders
        assert len(self.raw_folders) == len(self.compressed_folders), \
            f"Number of raw folders ({len(self.raw_folders)}) doesn't match compressed folders ({len(self.compressed_folders)})"
        
        # Initialize lists to store all image paths
        self.compressed_images = []
        self.raw_images = []
        
        # Collect all image paths from all folders
        for raw_folder, comp_folder in zip(self.raw_folders, self.compressed_folders):
            # Get images from this folder pair
            compressed_imgs = sorted(glob.glob(os.path.join(comp_folder, "*.png")))
            raw_imgs = sorted(glob.glob(os.path.join(raw_folder, "*.jpg"))) 
            
            # Verify matching number of images in this folder pair
            assert len(compressed_imgs) == len(raw_imgs), \
                f"Mismatch in images count for folders {raw_folder} ({len(raw_imgs)}) and {comp_folder} ({len(compressed_imgs)})"
            
            self.compressed_images.extend(compressed_imgs)
            self.raw_images.extend(raw_imgs)


    def __getitem__(self, idx):
        # load images and labels
        img_path = self.compressed_images[idx]
        label_path = self.raw_images[idx]
        
        img = Image.open(img_path).convert("RGB")
        label = Image.open(label_path).convert("RGB")
        
        
        image = np.array(img).astype(np.float32)
        image = np.transpose(image, axes=[2, 0, 1])
        image /= 255.0

        target = np.array(label).astype(np.float32)
        target = np.transpose(target, axes=[2, 0, 1])
        target /= 255.0
        
        return image, target

    def __len__(self):
        return len(self.compressed_images)
    