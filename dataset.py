from PIL import Image
from torchvision import datasets, transforms


def _norm_advprop(img):
    return img * 2.0 - 1.0


def build_train_transform(dest_image_size):
    normalize = transforms.Lambda(_norm_advprop)

    if not isinstance(dest_image_size, tuple):
        dest_image_size = (dest_image_size, dest_image_size)

    transform = transforms.Compose([
        transforms.Resize(dest_image_size, Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])

    return transform


def build_val_transform(dest_image_size):
    normalize = transforms.Lambda(_norm_advprop)

    if not isinstance(dest_image_size, tuple):
        dest_image_size = (dest_image_size, dest_image_size)

    transform = transforms.Compose([
        transforms.Resize(dest_image_size, Image.BICUBIC),
        transforms.ToTensor(),
        normalize
    ])

    return transform


def build_train_dataset(dest_image_size, data):
    transform = build_train_transform(dest_image_size)
    dataset = datasets.ImageFolder(data, transform)
    return dataset


def build_val_dataset(dest_image_size, data):
    transform = build_val_transform(dest_image_size)
    dataset = datasets.ImageFolder(data, transform)
    return dataset
