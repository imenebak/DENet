import os
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from image import *
# import torchvision.transforms.functional as F
import json

class listDataset(Dataset):
    def __init__(self, root, shape=None, shuffle=True, transform=None,  train=False, seen=0, batch_size=1, num_workers=4):
        if train:
            root = root *4
        random.shuffle(root)
        
        self.nSamples = len(root)
        self.lines = root
        self.transform = transform
        self.train = train
        self.shape = shape
        self.seen = seen
        self.batch_size = batch_size
        self.num_workers = num_workers
        # self.dataset=trainlist
        
        
    def __len__(self):
        return self.nSamples
    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        img_path = self.lines[index]
        img, target = load_data(img_path, self.train)
        
        gt_path = img_path.replace('.jpg', '.h5').replace('sequences', 'ground_truth')

        gt_file = h5py.File(gt_path, 'r')
        groundtruth = np.asarray(gt_file['density'])
        GT = np.sum(groundtruth)
        str1 = img_path.split('/sequences/')
        img_name = str1[1]
        img_name = img_name.encode('unicode-escape').decode('string_escape')

        file = open('./detection_result/train.txt', 'r')
        js = file.read()
        dict = json.loads(js)

        detection=dict[img_name]
        GT_detection=GT-detection
        target_sum=np.sum(target)
        if self.transform is not None:
            img = self.transform(img)
        return img,target,GT_detection,target_sum

        # return img, target
