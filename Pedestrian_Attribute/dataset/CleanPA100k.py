from torch.utils.data import Dataset
import torch
import os
from PIL import Image
import random
import hashlib
import pickle
from torchvision import transforms as T
from .augmentations import RandAugment
import numpy as np

class AttrDataset(Dataset):

    def __init__(self, split, transform=None, target_transform=None, testing=None,known_labels=None):

        # assert args.dataset in ['PETA', 'PETA_dataset', 'PA100k', 'RAP', 'RAP2'], \
        #     f'dataset name {args.dataset} is not exist'
        
            data_path = '/home/zhexuan_wh/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition-master/data/PA100k/dataset.pkl'
            # breakpoint()
            dataset_info = pickle.load(open(data_path, 'rb+'))

            img_id = dataset_info.image_name
            attr_label = dataset_info.label

            assert split in dataset_info.partition.keys(), f'split {split} is not exist'

            self.dataset = 'pa100k'
            self.transform = transform
            self.target_transform = target_transform
            self.testing = testing
            self.known_labels = known_labels
            self.root_path = '/home/zhexuan_wh/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition-master/data/PA100k/data'

            self.attr_id = dataset_info.attr_name
            self.attr_num = len(self.attr_id)

            self.img_idx = dataset_info.partition[split]

            if isinstance(self.img_idx, list):
                self.img_idx = self.img_idx[0]  # default partition 0
            self.img_num = self.img_idx.shape[0]
            self.img_id = [img_id[i] for i in self.img_idx]
            self.label = attr_label[self.img_idx]

    def __getitem__(self, index):

        imgname, gt_label = self.img_id[index], self.label[index]
        imgpath = os.path.join(self.root_path, imgname)
        img = Image.open(imgpath)

        if self.transform is not None:
            img = self.transform(img)

        gt_label = gt_label.astype(np.float32)

        # if self.target_transform is not None:
        #     gt_label = self.transform(gt_label)
        labels = torch.Tensor(gt_label).float()

        unk_mask_indices = get_unk_mask_indices(img, self.testing,self.attr_num,self.known_labels)
        
        mask = labels.clone()
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)
        sample = {}
        sample['image'] = img
        sample['labels'] = labels
        sample['mask'] = mask
        sample['imageIDs'] = imgname
        return sample

    def __len__(self):
        return len(self.img_id) // 1

def get_unk_mask_indices(image,testing,num_labels,known_labels,epoch=1):
    
    if testing:
        # for consistency across epochs and experiments, seed using hashed image array 
        random.seed(hashlib.sha1(np.array(image)).hexdigest())
        unk_mask_indices = random.sample(range(num_labels), (num_labels-int(known_labels)))
    else:
        # sample random number of known labels during training
        if known_labels>0:
            random.seed()
            num_known = random.randint(0,int(num_labels*0.75))
        else:
            num_known = 0

        unk_mask_indices = random.sample(range(num_labels), (num_labels-num_known))

    return unk_mask_indices

