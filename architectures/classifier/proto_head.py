"""
The metric-based protypical classifier (Nearest-Centroid Classifier) from ``Prototypical Networks for Few-shot Learning''.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import compute_prototypes


class PN_head(nn.Module):
    def __init__(self,
                 scale_cls: int =10.0, 
                 learn_scale: bool = True) -> None:
        super().__init__()
        if learn_scale:
            self.scale_cls = nn.Parameter(
                torch.FloatTensor(1).fill_(scale_cls), requires_grad=True
            )    
        else:
            self.scale_cls = scale_cls

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
        if query_images.dim() == 4:
            support_images = F.adaptive_avg_pool2d(support_images, 1).squeeze_(-1).squeeze_(-1)
            query_images = F.adaptive_avg_pool2d(query_images, 1).squeeze_(-1).squeeze_(-1)

        assert support_images.dim() == query_images.dim() == 2

            
        
        support_images = F.normalize(support_images, p=2, dim=1, eps=1e-12)
        query_images = F.normalize(query_images, p=2, dim=1, eps=1e-12)

        one_hot_label = F.one_hot(support_labels,num_classes = torch.max(support_labels).item()+1).float()

        #prototypes: [way, c]
        prototypes = compute_prototypes(support_images, one_hot_label)

        prototypes = F.normalize(prototypes, p=2, dim=1, eps=1e-12)

        classification_scores = self.scale_cls*torch.mm(query_images, prototypes.transpose(0, 1))

        return classification_scores

def create_model():
    return PN_head()