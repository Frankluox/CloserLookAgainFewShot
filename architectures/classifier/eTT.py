"""
Adapted from eTT: https://github.com/loadder/eTT_TMLR2022/blob/main
"""

import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import torch.nn as nn
from copy import deepcopy
import math
from .utils import prototype_scores, compute_prototypes, get_init_prefix



class PR(nn.Module):
    def __init__(self, init_prefix, in_dim=384, out_dim=64, warmup_teacher_temp=0.04, teacher_temp=0.04,
                 warmup_teacher_temp_epochs=None, nepochs=None, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        warmup_teacher_temp_epochs = 5 if nepochs > 5 else 2
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

        self.projector = nn.Linear(in_dim, out_dim)
        self.init_prefix = init_prefix

    def forward(self, prefix, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        prefix = F.normalize(prefix, dim=-1, p=2)
        init_prefix = F.normalize(self.init_prefix, dim=-1, p=2)

        prefix = self.projector(prefix)
        init_prefix = self.projector(init_prefix)

        temp = self.teacher_temp_schedule[epoch]
        prefix = prefix / self.student_temp
        init_out = F.softmax((init_prefix - self.center) / temp, dim=-1).detach()

        loss = torch.sum(-init_out * F.log_softmax(prefix, dim=-1), dim=-1).mean()

        self.update_center(init_out)
        return loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class eTTfinetuner(nn.Module):
    def __init__(self, backbone, ft_batchsize, feed_query_batchsize, ft_epoch,ft_lr_1,ft_lr_2):
        '''
        backbone: the pre-trained backbone
        ft_batchsize: batch size for finetune
        feed_query_batchsize: max number of query images processed once (avoid memory issues)
        ft_epoch: epoch of finetune
        ft_lr_1: backbone learning rate
        ft_lr_2: head learning rate
        '''
        super().__init__()
        self.ft_batchsize = ft_batchsize
        self.feed_query_batchsize = feed_query_batchsize
        self.ft_epoch = ft_epoch
        self.ft_lr_1 = ft_lr_1
        self.ft_lr_2 = ft_lr_2
        self.backbone = backbone


    def forward(self, query_images: Tensor, support_images: Tensor, support_labels) -> Tensor:
        """Take one task of few-shot support examples and query examples as input,
            output the logits of each query examples.

        Args:
            query_images: query examples. size: [num_query, c, h, w]
            support_images: support examples. size: [num_support, c, h, w]
            support_labels: labels of support examples. size: [num_support, way]
        Output:
            classification_scores: The calculated logits of query examples.
                                   size: [num_query, way]
        """
        support_size = support_images.size(0)

        device = support_images.device

        way = torch.max(support_labels).item()+1

        backbone = deepcopy(self.backbone)

        #[way,c]
        init_prefix = get_init_prefix(backbone, support_images, support_labels)

        # learnable parameters
        url_pa = []
        url_pa.append(torch.eye(backbone.outdim, backbone.outdim).to(device).requires_grad_(True))
        prefix_weight = init_prefix.clone().to(device).requires_grad_(True)
        

        #[way,c]->[way,l*2*c], every layer 2*way prefix 
        control_trans = nn.Sequential(
            nn.Linear(backbone.outdim, backbone.outdim//2),  # 1024 * 512
            nn.Tanh(),
            nn.Linear(backbone.outdim//2, backbone.n_layers * 2 * backbone.outdim)).requires_grad_(True).to(device)
        loss_fn = PR(init_prefix, nepochs=self.ft_epoch, in_dim=backbone.outdim).to(device)

        optim_list = [{'params': backbone.parameters()},
                      {'params': control_trans.parameters()},
                      {'params': prefix_weight},
                      {'params': loss_fn.parameters()}]
        
        set_optimizer_1 = torch.optim.SGD(optim_list, lr = self.ft_lr_1, momentum=0.9)
        set_optimizer_2 = torch.optim.SGD([{'params': url_pa[0]}], lr = self.ft_lr_2, momentum=0.9)


        backbone.eval()
        global_steps = self.ft_epoch*((support_size+self.ft_batchsize-1)//self.ft_batchsize)

        step = 0

        with torch.enable_grad():
            for epoch in range(self.ft_epoch):
                rand_id = np.random.permutation(support_size)
                for i in range(0, support_size , self.ft_batchsize):
                    lr_1 = 0.5 * self.ft_lr_1* (1. + math.cos(math.pi * step / global_steps))
                    lr_2 = 0.5 * self.ft_lr_2* (1. + math.cos(math.pi * step / global_steps))

                    for param_group in set_optimizer_1.param_groups:
                        param_group["lr"] = lr_1
                    set_optimizer_1.zero_grad()

                    for param_group in set_optimizer_2.param_groups:
                        param_group["lr"] = lr_2
                    set_optimizer_2.zero_grad()


                    #[way,n_layers,2,n_head,c=c/n_head]
                    prefix = control_trans(prefix_weight).view(way, backbone.n_layers, 2, backbone.n_head, backbone.outdim//backbone.n_head)
                    #[n_layers,2,n_head,way,c]
                    prefix = prefix.permute(1, 2, 3, 0, 4)
                    
                    #[2,n_layers,2,n_head,way,c]
                    prefix = prefix.unsqueeze(0).expand(2, -1, -1, -1, -1, -1)


                    selected_id = torch.from_numpy(rand_id[i: min(i+self.ft_batchsize, support_size)])
                    train_batch = support_images[selected_id]
                    label_batch = support_labels[selected_id] 
                    # [b,c]
                    train_batch = backbone(train_batch, prefix=prefix, return_feat=True)
                    train_batch = F.normalize(train_batch, p=2, dim=1, eps=1e-12)
                    train_batch = F.linear(train_batch, url_pa[0])

                    score = prototype_scores(train_batch, label_batch,
                                    train_batch)
                    loss = F.cross_entropy(score, label_batch)
                    dt_loss = loss_fn(prefix_weight, epoch)
                    loss += 0.1 * dt_loss

                    loss.backward()
                    set_optimizer_1.step()
                    set_optimizer_2.step()
                    step += 1

        backbone.eval()        

        prefix = control_trans(prefix_weight).view(way, backbone.n_layers, 2, backbone.n_head, backbone.outdim//backbone.n_head)
        prefix = prefix.permute(1, 2, 3, 0, 4)
        prefix = prefix.unsqueeze(0).expand(2, -1, -1, -1, -1, -1)

        query_runs = (query_images.size(0)+self.feed_query_batchsize-1)//self.feed_query_batchsize
        support_features = []
        query_features = []
        support_runs = (support_images.size(0)+self.feed_query_batchsize-1)//self.feed_query_batchsize

        for run in range(support_runs):
            support_features.append(backbone(support_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,query_images.size(0))], prefix=prefix, return_feat=True))
        for run in range(query_runs):
            query_features.append(backbone(query_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,query_images.size(0))], prefix=prefix, return_feat=True))
        support_features = torch.cat(support_features, dim=0)
        query_features = torch.cat(query_features, dim=0)
        support_features = F.linear(support_features, url_pa[0])
        query_features = F.linear(query_features, url_pa[0])

        classification_scores = prototype_scores(support_features, support_labels,
                        query_features)

        return classification_scores

def create_model(backbone, ft_batchsize, feed_query_batchsize, ft_epoch,ft_lr_1,ft_lr_2, mode = None):
    return eTTfinetuner(backbone, ft_batchsize, feed_query_batchsize, ft_epoch,ft_lr_1,ft_lr_2)