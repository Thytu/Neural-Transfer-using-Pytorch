"""
Handle 
"""

import torch
import torchvision.transforms as transforms

from PIL import Image
from numpy import ndarray


IMG_SIZE = 256 if torch.cuda.is_available() else 128
TENSOR_TO_PIL = transforms.ToPILImage()


def image_to_tensor(image, width=IMG_SIZE, height=IMG_SIZE, device='cpu') -> torch.tensor:
    """
    Convert a numpy array to a torch tensor

    :param image: numpy array
    :param width: width of the image
    :param height: height of the image
    :param device: device to use

    :return: torch tensor
    """

    width = width if width is not None else IMG_SIZE
    height = height if height is not None else IMG_SIZE

    image = Image.fromarray(image).convert('RGB').resize((width, height))

    compose = transforms.Compose([
        transforms.Resize((height, width)),
        transforms.ToTensor(),
    ])

    image = compose(image).unsqueeze(0)

    return image.to(device, torch.float)


def tensor_to_image(tensor: torch.tensor) -> ndarray:
    """
    Convert a torch tensor to a numpy array

    :param tensor: torch tensor

    :return: numpy array
    """

    image = TENSOR_TO_PIL(tensor.cpu().clone().squeeze(0))

    return image
