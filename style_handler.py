import torch
import torch.nn as nn
import torch.optim as optim

from typing import List
from normalization import Normalization
from loss import ContentLoss, StyleLoss

def create_noise_input(content_img, width=None, height=None, device='cpu'):
    """
    Create a noise input image based on the content image.

    :param content_img: The content image.
    :param width: The width of the noise image.
    :param height: The height of the noise image.
    :param device: The device to use.

    :return: The noise image.
    """

    width = width if width else content_img.shape[-1]
    height = height if height else content_img.shape[-2]

    return torch.randn((1, 3, height, width), device=device)

def _get_style_model_and_losses(
        cnn: nn.Module,
        style_img: torch.Tensor,
        content_img: torch.Tensor,
        content_layers: List[str] = ['conv_4'],
        style_layers: List[str] = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    ):
    """
    Get the style transfer model and the losses.

    :param cnn: The CNN to use.
    :param style_img: The style image.
    :param content_img: The content image.
    :param content_layers: The content layers.
    :param style_layers: The style layers.

    :return: The style transfer model and the losses.
    """

    # normalization module
    normalization = Normalization(cnn.norm_mean, cnn.norm_std, cnn.device).to(cnn.device)

    # just in order to have an iterable access to or list of content/syle losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    conv_nb = 0  # increment every time we see a conv

    allowed_layers = {
            nn.Conv2d: 'conv',
            nn.ReLU: 'relu',
            nn.MaxPool2d: 'pool',
            nn.BatchNorm2d: 'bn'
        }

    for layer in cnn._model.children():

        assert type(layer) in allowed_layers, f'Unrecognized layer: {layer.__class__.__name__}'

        if isinstance(layer, nn.Conv2d):
            conv_nb += 1
        elif isinstance(layer, nn.ReLU):
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place ones here.
            layer = nn.ReLU(inplace=False)

        name = f'{allowed_layers[type(layer)]}_{conv_nb}'

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach() # add content loss

            content_loss = ContentLoss(target)

            model.add_module(f"content_loss_{conv_nb}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach() # add style loss

            style_loss = StyleLoss(target_feature)

            model.add_module(f"style_loss_{conv_nb}", style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for conv_nb in range(len(model) - 1, -1, -1):
        if isinstance(model[conv_nb], ContentLoss) or isinstance(model[conv_nb], StyleLoss):
            break

    model = model[:(conv_nb + 1)]

    return model, style_losses, content_losses


def run_style_transfer(
        cnn: nn.Module,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        input_img: torch.Tensor,
        num_steps: int = 300,
        style_weight=1000000,
        content_weight=1,
        checkpoint_every=0
    ):
    """
    Run the style transfer.

    :param cnn: The CNN to use.
    :param content_img: The content image.
    :param style_img: The style image.
    :param input_img: The input image.
    :param num_steps: The number of steps to run the style transfer.
    :param style_weight: The style weight.
    :param content_weight: The content weight.
    :param checkpoint_every: The number of steps to run before saving a checkpoint.

    :return: The output image.
    """

    print('Building the style transfer model..')

    model, style_losses, content_losses = _get_style_model_and_losses(
        cnn,
        style_img, content_img,
        content_layers=['conv_4'], style_layers=['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
    )

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    model.requires_grad_(False)

    optimizer = optim.LBFGS([input_img])

    run = [0]
    best_image = [None]
    best_loss = [None]
    checkpoints = []

    print('Optimizing..')
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss

            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            if best_loss[0] is None or loss < best_loss[0]:
                best_loss[0] = loss
                best_image[0] = input_img.clone()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"run {run}:")
                print(
                    f'\tStyle Loss : {style_score.item():4f} Content Loss: {content_score.item():4f}',
                    f'\tBest Style Loss : {best_loss[0].item():4f}', sep='\n', end="\n\n"
                )

            if checkpoint_every > 0 and run[0] % checkpoint_every == 0:
                print("Saving checkpoint..")

                checkpoints.append(input_img.clone())

            return style_score + content_score

        optimizer.step(closure)

    print('Final correction...')
    with torch.no_grad():
        input_img[0].clamp_(0, 1)

    print("Done")
    return best_image[0], checkpoints
