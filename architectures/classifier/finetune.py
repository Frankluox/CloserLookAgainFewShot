"""
A unified implementation of gradient-based adaptation-time classifiers,
including finetune, URL and cosine classifer.
"""
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
import torch.nn as nn
from copy import deepcopy
import math
from .utils import CC_head, prototype_scores




class FinetuneModel(torch.nn.Module):
    """
    the overall finetune module that incorporates a backbone and a head.
    """
    def __init__(self, backbone, way, device, use_alpha, use_beta, head):
        super().__init__()
        '''
        backbone: the pre-trained backbone
        way: number of classes
        device: GPU ID
        use_alpha: for TSA only. Whether use adapter to adapt the backbone.
        use_beta: for URL and TSA only. Whether use  pre-classifier transformation.
        head: Use fc head or PN head to finetune.
        '''
        self.backbone = deepcopy(backbone).to(device)
        if head == "cc":
            self.L = CC_head(backbone.outdim, way).to(device)
        elif head == "fc":
            self.L = nn.Linear(backbone.outdim, way).to(device)
            self.L.weight.data.fill_(1)
            self.L.bias.data.fill_(0)
        self.use_beta = use_beta
        self.head = head

    def forward(self, x, backbone_grad = True):
        # turn backbone_grad off if backbone is not to be finetuned
        if backbone_grad:
            x = self.backbone(x)
        else:
            with torch.no_grad():
                x = self.backbone(x)

        if self.head == "NCC" and self.use_beta:
            x = self.backbone.beta(x)
        if x.dim() == 4:
            x = F.adaptive_avg_pool2d(x, 1).squeeze_(-1).squeeze_(-1)
        x = F.normalize(x, dim=1)
        if not self.head == "NCC":
            x = self.L(x)
        return x

class Finetuner(nn.Module):
    def __init__(self, backbone, ft_batchsize, feed_query_batchsize, ft_epoch,ft_lr_1,ft_lr_2, use_alpha, use_beta, head = "fc"):
        '''
        backbone: the pre-trained backbone
        ft_batchsize: batch size for finetune
        feed_query_batchsize: max number of query images processed once (avoid memory issues)
        ft_epoch: epoch of finetune
        ft_lr_1: backbone learning rate
        ft_lr_2: head learning rate
        use_alpha: for TSA only. Whether use adapter to adapt the backbone.
        use_beta: for URL and TSA only. Whether use  pre-classifier transformation.
        head: the classification head--"fc", "NCC" or "cc"
        '''
        super().__init__()
        self.ft_batchsize = ft_batchsize
        self.feed_query_batchsize = feed_query_batchsize
        self.ft_epoch = ft_epoch
        self.ft_lr_1 = ft_lr_1
        self.ft_lr_2 = ft_lr_2
        self.use_alpha = use_alpha
        self.use_beta = use_beta
        self.head =  head
        self.backbone = backbone

        assert head in ["fc", "NCC", "cc"]

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
        model = FinetuneModel(self.backbone, way, device, self.use_alpha, self.use_beta, self.head)

        # By default, SGD is adopted as the optimizer. Other optimizers like Adam can be used as well.

        if self.ft_lr_1==0.:
            # URL or Linear classifier or cosine classifier. No backbone tuning needed.
            set_optimizer_1 = None
        elif self.head == "NCC" and self.use_alpha:
            # if using adapter, then the adapters are tuned
            alpha_params = [v for k, v in model.backbone.named_parameters() if 'alpha' in k]
            set_optimizer_1 = torch.optim.SGD(alpha_params, lr = self.ft_lr_1, momentum=0.9)
        else:
            set_optimizer_1 = torch.optim.SGD(model.backbone.parameters(), lr = self.ft_lr_1, momentum=0.9)
        
        if self.head == "NCC" and self.use_beta:
            beta_params = [v for k, v in model.backbone.named_parameters() if 'beta' in k]
            set_optimizer_2 = torch.optim.SGD(beta_params, lr = self.ft_lr_2, momentum=0.9)
        elif not self.head == "NCC":
            set_optimizer_2 = torch.optim.SGD(model.L.parameters(), lr = self.ft_lr_2, momentum=0.9)
        else:
            # If using NCC as the tuning loss and no pre-classifier transformation is used,
            # then there is no head to be learned.
            set_optimizer_2 = None

        model.eval()

        # total finetuning steps
        global_steps = self.ft_epoch*((support_size+self.ft_batchsize-1)//self.ft_batchsize)

        step = 0

        with torch.enable_grad():
            for epoch in range(self.ft_epoch):
                # randomly suffule support set
                rand_id = np.random.permutation(support_size)

                for i in range(0, support_size , self.ft_batchsize):
                    # by default, cosine LR shedule is used.
                    lr_1 = 0.5 * self.ft_lr_1* (1. + math.cos(math.pi * step / global_steps))
                    lr_2 = 0.5 * self.ft_lr_2* (1. + math.cos(math.pi * step / global_steps))
                    if set_optimizer_1 is not None:
                        for param_group in set_optimizer_1.param_groups:
                            param_group["lr"] = lr_1
                        set_optimizer_1.zero_grad()
                    if set_optimizer_2 is not None:
                        for param_group in set_optimizer_2.param_groups:
                            param_group["lr"] = lr_2
                        set_optimizer_2.zero_grad()


                    selected_id = torch.from_numpy(rand_id[i: min(i+self.ft_batchsize, support_size)])
                    train_batch = support_images[selected_id]
                    label_batch = support_labels[selected_id] 

                    if set_optimizer_1 is not None:
                        train_batch = model(train_batch)
                    else:
                        train_batch = model(train_batch, backbone_grad = False)

                    if not self.head == "NCC":
                        loss = F.cross_entropy(train_batch, label_batch)
                    else:
                        score = prototype_scores(train_batch, label_batch,
                                       train_batch)
                        loss = F.cross_entropy(score, label_batch)

                    loss.backward()
                    if set_optimizer_1 is not None:
                        set_optimizer_1.step()
                    if set_optimizer_2 is not None:
                        set_optimizer_2.step()
                    step += 1

        model.eval()            

        # number of feed-forward calculations to calculate all query embeddings
        query_runs = (query_images.size(0)+self.feed_query_batchsize-1)//self.feed_query_batchsize

        if not self.head == "NCC":
            scores = []
            for run in range(query_runs):
                # for non-NCC head, the model directly ouputs score
                scores.append(model(query_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,query_images.size(0))]))
            classification_scores = torch.cat(scores, dim=0)
        else:
            support_features = []
            query_features = []
            # number of feed-forward calculations to calculate all support embeddings
            support_runs = (support_images.size(0)+self.feed_query_batchsize-1)//self.feed_query_batchsize
            for run in range(support_runs):
                support_features.append(model(support_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,support_images.size(0))]))
            for run in range(query_runs):
                query_features.append(model(query_images[run*self.feed_query_batchsize:min((run+1)*self.feed_query_batchsize,query_images.size(0))]))
            support_features = torch.cat(support_features, dim=0)
            query_features = torch.cat(query_features, dim=0)
            classification_scores = prototype_scores(support_features, support_labels,
                            query_features)

        return classification_scores

def create_model(backbone, ft_batchsize, feed_query_batchsize, ft_epoch,ft_lr_1,ft_lr_2, use_alpha, use_beta, head = 'fc'):
    return Finetuner(backbone, ft_batchsize, feed_query_batchsize, ft_epoch,ft_lr_1,ft_lr_2, use_alpha, use_beta, head)