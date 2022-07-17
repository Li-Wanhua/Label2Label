import torchvision.transforms as transforms
from dataset.cocodataset import CoCoDataset, LFWADataset, ClothDataset
from dataset.AttrDataset import AttrDataset
from utils.cutout import SLCutoutPIL
from RandAugment import RandAugment
import os.path as osp

def get_datasets(args):
    if args.orid_norm:
        normalize = transforms.Normalize(mean=[0, 0, 0],
                                         std=[1, 1, 1])
        # print("mean=[0, 0, 0], std=[1, 1, 1]")
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        # print("mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    train_data_transform_list = [transforms.Resize((args.img_size, args.img_size)),
                                            RandAugment(2,9),
                                               transforms.ToTensor(),
                                               normalize]
    try:
        # for q2l_infer scripts
        if args.cutout:
            print("Using Cutout!!!")
            train_data_transform_list.insert(1, SLCutoutPIL(n_holes=args.n_holes, length=args.length))
    except Exception as e:
        Warning(e)
    train_data_transform = transforms.Compose(train_data_transform_list)

    test_data_transform = transforms.Compose([
                                            transforms.Resize((args.img_size, args.img_size)),
                                            transforms.ToTensor(),
                                            normalize])
    

    if args.dataname == 'coco' or args.dataname == 'coco14':
        # ! config your data path here.
        dataset_dir = args.dataset_dir
        train_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'train2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_train2014.json'),
            input_transform=train_data_transform,
            labels_path='data/coco/train_label_vectors_coco14.npy',
        )
        val_dataset = CoCoDataset(
            image_dir=osp.join(dataset_dir, 'val2014'),
            anno_path=osp.join(dataset_dir, 'annotations/instances_val2014.json'),
            input_transform=test_data_transform,
            labels_path='data/coco/val_label_vectors_coco14.npy',
        )    
    elif args.dataname == 'lfwa':
        train_dataset = LFWADataset(
            '/home/zhexuan_wh/dataset/LFWA/lfw',
            '/home/zhexuan_wh/PycharmProjects/face/train_lfwa.txt',
            train_data_transform, 250, 250
        )
        val_dataset = LFWADataset(
            '/home/zhexuan_wh/dataset/LFWA/lfw',
            '/home/zhexuan_wh/PycharmProjects/face/val_lfwa.txt',
            test_data_transform, 250, 250
        )
    elif args.dataname == 'cloth':
        a = transforms.Resize((289,289))
        normalize = transforms.Normalize(mean=[0.490, 0.463, 0.434],
                                        std=[1, 1, 1])
    
        transform_train = transforms.Compose([
                a,
                transforms.RandomCrop((256,256)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        transform_train.transforms.insert(0, RandAugment(2, 9))
        train_dataset = ClothDataset(
            '/home/zhexuan_wh/dataset/ClothingAttributeDataset/images',
            '/home/zhexuan_wh/dataset/ClothingAttributeDataset/data.pkl',
            transform_train, 256, 256
        )
        val_dataset = ClothDataset(
            '/home/zhexuan_wh/dataset/ClothingAttributeDataset/images',
            '/home/zhexuan_wh/dataset/ClothingAttributeDataset/data.pkl',
            transforms.Compose([
                a,
                transforms.CenterCrop((256,256)),
                transforms.ToTensor(),
                normalize
            ]), 256, 256, False
        )
    else:
        raise NotImplementedError("Unknown dataname %s" % args.dataname)

    print("len(train_dataset):", len(train_dataset)) 
    print("len(val_dataset):", len(val_dataset))
    return train_dataset, val_dataset
