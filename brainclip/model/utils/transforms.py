import torchio as tio
from torch import Tensor

def get_transforms_list():
    return [
        tio.RescaleIntensity((0, 1)),
        tio.RandomAffine(
            scales=(0.95, 1.05),
            degrees=(-6, 6),
            translation=(-5, 5),
            isotropic=False,
            center='image',
            default_pad_value='minimum',
            image_interpolation='linear',
            label_interpolation='nearest',
            check_shape=True,
            p=0.8
        ),
        tio.RandomNoise(
            mean=300,
            std=(0.005, 0.01),
            p=0.5
        ),
        tio.RandomFlip(axes='lr', flip_probability=0.5),
    ]


def get_transforms():
    return tio.Compose(
            get_transforms_list()
        )
    

def apply_transform(img:Tensor, transforms:tio.Compose):
    img = tio.ScalarImage(tensor=img)
    transformed_img = transforms(img)["data"]
    return transformed_img
