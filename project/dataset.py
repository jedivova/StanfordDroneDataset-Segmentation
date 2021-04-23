import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from dataset_SDD import SDD_dataset, CLASSES
from dataset_UAVid import UAVid_dataset
from transforms import pre_transforms, hard_transforms, get_preprocessing
import albumentations as albu


def get_loaders(preprocessing_fn, batch_size=12, num_workers=4):
    preprocessing = get_preprocessing(preprocessing_fn)


    SDD_train_PATH = r'Z:\test tasks\Sber_robo\SDD_images\imgs\train'
    SDD_val_PATH = r'Z:\test tasks\Sber_robo\SDD_images\imgs\test'

    train_transforms = albu.Compose([pre_transforms(image_size=512),
                                     hard_transforms(crop_size=512)])
    valid_transforms = pre_transforms(image_size=512)
    SDD_train = SDD_dataset(SDD_train_PATH, augmentations=train_transforms, preprocessing=preprocessing)
    SDD_val = SDD_dataset(SDD_val_PATH, augmentations=valid_transforms, preprocessing=preprocessing)

    #############

    UAVid_train_PATH = r'Z:\test tasks\Sber_robo\uavid_v1.5_official_release_image\uavid_train'
    UAVid_val_PATH = r'Z:\test tasks\Sber_robo\uavid_v1.5_official_release_image\uavid_val'

    train_transforms = albu.Compose([pre_transforms(image_size=1024),
                                     hard_transforms(crop_size=512)])
    valid_transforms = albu.Compose([pre_transforms(image_size=1024),
                                     albu.CropNonEmptyMaskIfExists(512, 512, p=1)])
    UAVid_train = UAVid_dataset(UAVid_train_PATH, augmentations=train_transforms, preprocessing=preprocessing)
    UAVid_val = UAVid_dataset(UAVid_val_PATH, augmentations=valid_transforms, preprocessing=preprocessing)

    train_dataset = SDD_train + UAVid_train
    val_dataset = SDD_val + UAVid_val
    print(f"found {len(SDD_train)} images for SDD")
    print(f"found {len(UAVid_train)} images for UAVid")
    SDD_weight = 1 / len(SDD_train)
    UAVid_weight = 1 / len(UAVid_train)
    samples_weight = np.array([SDD_weight] * len(SDD_train) + [UAVid_weight] * len(UAVid_train))
    sampler = WeightedRandomSampler(torch.from_numpy(samples_weight), len(samples_weight))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    return loaders


if __name__=='__main__':
    import segmentation_models_pytorch as smp
    preprocessing_fn = smp.encoders.get_preprocessing_fn('mobilenet_v2', 'imagenet')
    loaders = get_loaders(preprocessing_fn, batch_size=12, num_workers=4)
    x,y = next(iter(loaders["valid"]))
    print(x.shape, y.shape)
