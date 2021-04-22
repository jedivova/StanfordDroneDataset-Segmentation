from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
from pathlib import Path
import numpy as np
from transforms import train_transforms, valid_transforms, get_preprocessing
import pandas as pd


CLASSES = {'restricted_area': 0, 'road': 1, 'building': 2, 'human': 3, 'car': 4}

class SDD_dataset(Dataset):
    def __init__(self, data_path, augmentations=None, preprocessing=None) -> None:
        self.augmentations = augmentations
        self.preprocessing = preprocessing
        self.class_num = len(CLASSES)
        self.bbox_ann = pd.read_csv(r'Z:\test tasks\Sber_robo\SDD_images\new_anns.csv', delimiter=',')
        self.markers_ann = pd.read_csv(r'Z:\test tasks\Sber_robo\SDD_images\markers_ann.csv', delimiter=',')
        self.IMG_paths = list(Path(data_path).glob('*.jpg'))


    def __len__(self) -> int:
        return len(self.IMG_paths)


    def get_labels(self, img_p):
        img = cv2.imread(str(img_p))[:, :, ::-1]
        marker_p = self.markers_ann[self.markers_ann['img_name'] == img_p.name]['markers'].item()
        markers = cv2.imread(marker_p, 0).astype(np.int32)
        labels = cv2.watershed(img, markers) - 1  # cause watershed label idxs starts from 1

        elems = self.bbox_ann[self.bbox_ann['img_name'] == img_p.name]
        for index, row in elems.iterrows():
            label_idx = CLASSES[row['class_name']]
            xmin, ymin, xmax, ymax = row[1], row[2], row[3], row[4]

            labels[ymin:ymax, xmin:xmax] = label_idx
        return img, labels


    def __getitem__(self, idx: int):
        image, labels = self.get_labels(self.IMG_paths[idx])

        masks = [(labels == v) for v in range(self.class_num)]
        mask = np.stack(masks, axis=-1).astype('float')

        result = {"image": image, "mask": mask}

        if self.augmentations is not None:
            result = self.augmentations(**result)

        # apply preprocessing
        if self.preprocessing:
            result = self.preprocessing(**result)

        return result['image'], result['mask']


def get_loaders(preprocessing_fn, batch_size=12, num_workers=4) -> dict:
    Train_PATH = r'Z:\test tasks\Sber_robo\SDD_images\imgs\train'
    Val_PATH = r'Z:\test tasks\Sber_robo\SDD_images\imgs\test'

    train_dataset = SDD_dataset(Train_PATH, augmentations=train_transforms, preprocessing=get_preprocessing(preprocessing_fn))
    valid_dataset = SDD_dataset(Val_PATH, augmentations=valid_transforms, preprocessing=get_preprocessing(preprocessing_fn))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    loaders = {
        "train": train_loader,
        "valid": valid_loader
    }

    return loaders


if __name__=='__main__':
    SDD_dataset('./')