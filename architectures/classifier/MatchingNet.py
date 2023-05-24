"""
MatchingNet classifier head. 
Adapted from https://github.com/nupurkmr9/S2M2_fewshot/blob/master/methods/matchingnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from .utils import compute_prototypes

class MN_head(nn.Module):

    def __init__(
        self) -> None:
        super().__init__()


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
            query_images = F.adaptive_avg_pool2d(query_images, 1).squeeze_(-1).squeeze_(-1)
            support_images = F.adaptive_avg_pool2d(support_images, 1).squeeze_(-1).squeeze_(-1)

        assert support_images.dim() == query_images.dim() == 2

        support_images = F.normalize(support_images, p=2, dim=1, eps=1e-12)
        query_images = F.normalize(query_images, p=2, dim=1, eps=1e-12)

        one_hot_label = F.one_hot(support_labels,num_classes = torch.max(support_labels).item()+1).float()

        #[num_support, num_query]
        scores = support_images.mm(query_images.transpose(0,1))

        #[num_query, n_way]
        classification_scores = compute_prototypes(scores, one_hot_label).transpose(0,1)


        # The original paper use cosine simlarity, but here we scale it by 100 to strengthen
        # highest probability after softmax
        classification_scores = F.relu(classification_scores) * 100
        classification_scores = F.softmax(classification_scores, dim=1)

        return classification_scores


def create_model():
    return MN_head()
