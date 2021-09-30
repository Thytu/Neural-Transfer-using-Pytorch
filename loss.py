"""
TODO
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ContentLoss(nn.Module):
    """
    Measures how different the content is between two image.

    Important detail: although this module is named ContentLoss, it is not a true PyTorch Loss function.
    If you want to define your content loss as a PyTorch Loss function, you have to create a PyTorch autograd function to recompute/implement the gradient manually in the backward method.
    """

    def __init__(self, target,):
        super(ContentLoss, self).__init__()

        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)

        return input

def gram_matrix(input):
    batch_size, nb_feature_maps, c, d = input.size()
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(batch_size * nb_feature_maps, c * d)  # resise F_XL into \hat F_XL

    G = torch.mm(features, features.t())  # compute the gram product

    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(batch_size * nb_feature_maps * c * d)


class StyleLoss(nn.Module):
    """
    Measures how different the style is between two images.

    Important detail: although this module is named ContentLoss, it is not a true PyTorch Loss function.
    If you want to define your content loss as a PyTorch Loss function, you have to create a PyTorch autograd function to recompute/implement the gradient manually in the backward method.
    """

    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()

        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)

        return input
