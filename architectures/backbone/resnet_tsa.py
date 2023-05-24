# mostly copy-paste from TSA: https://github.com/VICO-UoE/URL/blob/master/models/tsa.py

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import math
import copy
import torch.nn.functional as F



class conv_tsa(nn.Module):
    def __init__(self, orig_conv):
        super(conv_tsa, self).__init__()
        # the original conv layer
        self.conv = copy.deepcopy(orig_conv)
        self.conv.weight.requires_grad = False
        planes, in_planes, _, _ = self.conv.weight.size()
        stride, _ = self.conv.stride
        # task-specific adapters

        # alpha in original paper
        self.alpha = nn.Parameter(torch.ones(planes, in_planes, 1, 1))
        self.alpha.requires_grad = True

        # channel-wise mulplication
        # if planes == in_planes:
        #     self.tsa = True
        #     self.alpha = nn.Parameter(torch.zeros(planes))
        #     self.alpha.requires_grad = True
        # else:
        #     self.tsa = False

    def forward(self, x):
        y = self.conv(x)
        

        # alpha in original paper
        # residual adaptation in matrix form
        y = y + F.conv2d(x, self.alpha, stride=self.conv.stride)

        # residual adaptation in channel-wise form
        # channel-wise mulplication
        # if self.tsa:
        #     alpha = torch.unsqueeze(torch.unsqueeze(torch.unsqueeze(self.alpha,-1),-1),0)
        #     y = y + x*alpha
        return y


class pa(nn.Module):
    """ 
    pre-classifier alignment (PA) mapping from 'Universal Representation Learning from Multiple Domains for Few-shot Classification'
    (https://arxiv.org/pdf/2103.13841.pdf)
    """
    def __init__(self, feat_dim):
        super(pa, self).__init__()
        # define pre-classifier alignment mapping
        self.weight = nn.Parameter(torch.ones(feat_dim, feat_dim, 1, 1))

        

        self.weight.requires_grad = True

    def forward(self, x):
        if len(list(x.size())) == 2:
            x = x.unsqueeze(-1).unsqueeze(-1)
        x = F.conv2d(x, self.weight.to(x.device))
        return x

class resnet_tsa(nn.Module):
    """ Attaching task-specific adapters (alpha) and/or PA (beta) to the ResNet backbone """
    def __init__(self, orig_resnet):
        super(resnet_tsa, self).__init__()
        # freeze the pretrained backbone
        for k, v in orig_resnet.named_parameters():
                v.requires_grad=False

        # attaching task-specific adapters (alpha) to each convolutional layers
        # note that we only attach adapters to residual blocks in the ResNet
        # for block in orig_resnet.layer1:
        block = orig_resnet.layer1
        for name, m in block.named_children():
            if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                new_conv = conv_tsa(m)
                setattr(block, name, new_conv)

        # for block in orig_resnet.layer2:
        block = orig_resnet.layer2
        for name, m in block.named_children():
            if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                new_conv = conv_tsa(m)
                setattr(block, name, new_conv)

        block = orig_resnet.layer3
        # for block in orig_resnet.layer3:
        for name, m in block.named_children():
            if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                new_conv = conv_tsa(m)
                setattr(block, name, new_conv)

        block = orig_resnet.layer4
        # for block in orig_resnet.layer4:
        for name, m in block.named_children():
            if isinstance(m, nn.Conv2d) and m.kernel_size[0] == 3:
                new_conv = conv_tsa(m)
                setattr(block, name, new_conv)

        self.backbone = orig_resnet

        # attach pre-classifier alignment mapping (beta)
        feat_dim = orig_resnet.outdim
        beta = pa(feat_dim)
        setattr(self, 'beta', beta)
        self.reset()

    def forward(self, x):
        return self.backbone.forward(x=x)

    def get_state_dict(self):
        """Outputs all the state elements"""
        return self.backbone.state_dict()

    def get_parameters(self):
        """Outputs all the parameters"""
        return [v for k, v in self.backbone.named_parameters()]

    def reset(self):
        # initialize task-specific adapters (alpha)
        for k, v in self.backbone.named_parameters():
            if 'alpha' in k:
                v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0
                # v.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device) * 0.0001

        # initialize pre-classifier alignment mapping (beta)
        v = self.beta.weight
        self.beta.weight.data = torch.eye(v.size(0), v.size(1)).unsqueeze(-1).unsqueeze(-1).to(v.device)


def create_model(backbone):
    return resnet_tsa(backbone)