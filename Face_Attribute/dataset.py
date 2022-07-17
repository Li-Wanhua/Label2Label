from torch.utils.data import Dataset
import torch
import os
from PIL import Image
from torchvision import transforms
from augmentations import RandAugment
from loguru import logger


class LFWADataset(Dataset):
    def __init__(self, img_root, target_path, transform, height, width):
        super(LFWADataset, self).__init__()
        
        self.target_list = []
        self.img_root = img_root
        self.img_paths = []
        self.transform = transform
        self.height = height
        self.width = width
        
        with open(target_path) as f:
            data = f.read().split('\n')
        for img_target in data:
            img_target = img_target.split()
            self.img_paths.append(img_target[0])
            target = img_target[1:41]
            target = [int(x) for x in target]
            self.target_list.append(target) 
        logger.info(f'all images: {len(self.img_paths)}')
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_root, self.img_paths[index]))
        img = img.convert('RGB')
        
        img = img.resize((self.height, self.width),Image.ANTIALIAS)
        img = self.transform(img)
        target = torch.tensor(self.target_list[index]).float()
        target = torch.where(target>0, torch.ones_like(target), torch.zeros_like(target))
        return img, target
    def __len__(self):
        return len(self.img_paths)


def create_dataloader(args):
    if args.dataset == 'LFWA':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
        if args.randAug:
            logger.warning('Use Rand Augmentation', args.n, args.m)
            transform_train.transforms.insert(0, RandAugment(args.n, args.m))
        train_dataset = LFWADataset(
            args.data_root,
            args.train_target,
            transform_train, args.height, args.width
        )
        val_dataset = LFWADataset(
                args.data_root,
                args.val_target,
                transforms.Compose([
                    transforms.ToTensor(),
                    normalize,
                ]), args.height, args.width
            ) 
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size, shuffle=False, #Always False
        num_workers=args.workers, pin_memory=True)
    return train_loader, val_loader
