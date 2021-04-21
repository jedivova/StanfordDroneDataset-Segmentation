import albumentations as albu
import cv2
from albumentations.pytorch import ToTensor

BORDER_CONSTANT = 0
BORDER_REFLECT = 2

def pre_transforms(image_size=512):
    return albu.Compose([
        albu.LongestMaxSize(max_size=image_size, always_apply=True),
        albu.PadIfNeeded(image_size, image_size, border_mode=BORDER_CONSTANT, value=0),
    ])


def hard_transforms(crop_size=320):
    return albu.Compose([
        albu.ShiftScaleRotate(shift_limit=0, scale_limit=0.1, rotate_limit=180, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
        albu.CropNonEmptyMaskIfExists(crop_size, crop_size, p=1),
        albu.HueSaturationValue(p=0.3),
        albu.OneOf([
            albu.IAAAdditiveGaussianNoise(),
            albu.GaussNoise(),
            albu.MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=1)
        ], p=0.3),
        albu.OneOf([
            albu.MotionBlur(p=0.2),
            albu.MedianBlur(blur_limit=3, p=0.1),
            albu.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        albu.OneOf([
            albu.CLAHE(clip_limit=2),
            albu.IAASharpen(),
            albu.IAAEmboss(),
            albu.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            albu.RandomGamma(gamma_limit=(85, 115), p=0.3),
        ], p=0.3),
        albu.JpegCompression(quality_lower=80, quality_upper=100, p=0.5),
        albu.Cutout(num_holes=25, max_h_size=5, max_w_size=5, fill_value=0, p=0.3),
        ], p=1)

def post_transforms():
    return albu.Compose([
        albu.CenterCrop(height=512, width=512, p=1)],
#         albu.Normalize(),
#         ToTensor()],
        p=1)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


train_transforms = albu.Compose([
    pre_transforms(),
    hard_transforms()
])

valid_transforms = albu.Compose([
    pre_transforms()
])

# show_transforms = albu.Compose([pre_transforms(), hard_transforms()])