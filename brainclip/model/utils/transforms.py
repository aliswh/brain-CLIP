import torchio as tio
from torch import Tensor

def get_transforms_list():
    return [
        #tio.RescaleIntensity((0, 1)),
        tio.RandomAffine(
            scales=(0.98, 1.02),
            degrees=(-3, 3),
            translation=(-3, 3),
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
        #tio.RandomBiasField(coefficients=0.1, p=0.5),
        #tio.RandomBlur(p=0.5),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5)
    ]


def get_transforms():
    return tio.Compose(
            get_transforms_list()
        )
    

def apply_transform(img:Tensor, transforms:tio.Compose):
    img = tio.ScalarImage(tensor=img)
    transformed_img = transforms(img)["data"]
    return transformed_img
