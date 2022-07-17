import os
import pickle
import numpy as np
import torch.utils.data as data
from PIL import Image
from .augmentations import RandAugment
import torchvision.transforms as T


def get_pkl_rootpath(dataset):
    root = os.path.join("./data", f"{dataset}")
    data_path = os.path.join(root, 'dataset.pkl')

    return data_path

class AttrDataset(data.Dataset):

    def __init__(self, split, args, transform=None, target_transform=None):

        # assert args.dataset in ['PETA', 'PETA_dataset', 'PA100k', 'RAP', 'RAP2'], \
        #     f'dataset name {args.dataset} is not exist'
        if args.dataset == 'RAP':
        # if 1:
            if split == 'test':
                path = '/home/zhexuan_wh/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition-master/data/rap/test.txt'
            else:
                path = '/home/zhexuan_wh/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition-master/data/rap/train.txt'
            with open(path) as f:
                lines = f.read().split('\n')
                lines = lines[0: len(lines) - 1]

            img_id = []
            label = []
            for line in lines:
                index = line.find('png')
                img_id.append(line[0:index+3])
                l = line[index+3: len(line)].split()
                l = np.array([int(x) for x in l]).astype('uint8')
                label.append(l)
            self.img_id = img_id
            self.label = np.array(label) 
            self.transform = transform
            self.target_transform = target_transform
            self.attr_num = 51
            self.dataset = 'RAP'
            self.root_path = '/home/zhexuan_wh/PycharmProjects/Strong_Baseline_of_Pedestrian_Attribute_Recognition-master/data/rap/images'
        else:
            data_path = get_pkl_rootpath(args.dataset)

            dataset_info = pickle.load(open(data_path, 'rb+'))

            img_id = dataset_info.image_name
            attr_label = dataset_info.label

            assert split in dataset_info.partition.keys(), f'split {split} is not exist'

            self.dataset = args.dataset
            self.transform = transform
            self.target_transform = target_transform

            self.root_path = dataset_info.root

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

        if self.target_transform is not None:
            gt_label = self.transform(gt_label)

        return img, gt_label, imgname

    def __len__(self):
        return len(self.img_id)


def get_transform(args):
    height = args.height
    width = args.width
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = T.Compose([
        T.Resize((height, width)),
        T.Pad(10),
        T.RandomCrop((height, width)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        normalize,
    ])
    if args.randAug:
        print('Use Rand Augmentation', args.n, args.m)
        train_transform.transforms.insert(1, RandAugment(args.n, args.m))
        

    valid_transform = T.Compose([
        T.Resize((height, width)),
        T.ToTensor(),
        normalize
    ])

    return train_transform, valid_transform

if __name__ == '__main__':
    pass
    # d = AttrDataset('test', None)