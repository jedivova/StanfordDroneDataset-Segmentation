from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
from pathlib import Path
import numpy as np
from transforms import train_transforms, valid_transforms, get_preprocessing

class UAVid_dataset(Dataset):
    CLASSES = ['Clutter', 'Building', 'Road', 'Car', 'Tree', 'Vegetation', 'Human']

    def __init__(self, data_path, augmentations=None, preprocessing=None) -> None:
        self.augmentations = augmentations
        self.preprocessing = preprocessing
        self.class_num = 7
        self.dataset = self.get_data_paths(data_path)

    def get_data_paths(self, dataset_path):
        dataset = []
        img_list = list(Path(dataset_path).glob('seq*\Images\*.png'))
        print(f'found {len(img_list)} images')
        for img_path in img_list:
            mask_path = list(img_path.parents)[1].joinpath('Masks', img_path.name)
            dataset.append({'img_path': img_path, 'mask_path': mask_path})
        return dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict:
        image = cv2.imread(str(self.dataset[idx]['img_path']))[:, :, ::-1]
        mask = cv2.imread(str(self.dataset[idx]['mask_path']))[:, :, 0]

        masks = [(mask == v) for v in range(self.class_num)]
        mask = np.stack(masks, axis=-1).astype('float')

        result = {"image": image, "mask": mask}

        if self.augmentations is not None:
            result = self.augmentations(**result)

        # apply preprocessing
        if self.preprocessing:
            result = self.preprocessing(**result)

        return result['image'], result['mask']


def get_val_batches(valid_loader):
    '''
    We should validate model on the same images through epoches.
    :return: list of batches
    '''
    Val_batches = []
    for i in tqdm(range(10)):
        for image, mask in tqdm(valid_loader):
            Val_batches.append([image, mask])
    return Val_batches

def get_loaders(preprocessing_fn, batch_size=12, num_workers=4):
    Train_PATH = r'Q:\Downloads\uavid_v1.5_official_release_image\uavid_train'
    Val_PATH = r'Q:\Downloads\uavid_v1.5_official_release_image\uavid_val'

    train_dataset = UAVid_dataset(Train_PATH, augmentations=train_transforms, preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = UAVid_dataset(Val_PATH, augmentations=valid_transforms, preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    # valid_loader = get_val_batches(valid_loader)


    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    return loaders


if __name__=='__main__':
    UAVid_dataset('./')