"""
TODO
"""

import torch
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

IMG_SIZE = 512 if torch.cuda.is_available() else 128
COMPOSED_TRANSFORMERS = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
])

def load_image_to_tensor(image_name, width=IMG_SIZE, height=IMG_SIZE, device='cpu') -> torch.tensor:
    image = Image.open(image_name)

    if image_name.endswith('.png'):
        image = image.convert('RGB')

    if width is not None and height is not None:
        image = image.resize((width, height))

    image = COMPOSED_TRANSFORMERS(image).unsqueeze(0)

    return image.to(device, torch.float)


def show_tensor_image(image: torch.Tensor, title=None):
    plt.figure()

    plt.imshow(transforms.ToPILImage()(image.cpu().clone().squeeze(0)))
    if title is not None:
        plt.title(title)

    plt.show()
