"""
Logistic Regression classifier head
"""
from torch import nn
from torch.nn import functional as F
import torch
from sklearn.linear_model import LogisticRegression
from torch import Tensor

class Logistic_Regression(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, query_images: Tensor, support_images: Tensor, 
                support_labels):
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

        if support_images.dim() == 4:
            support_images = F.adaptive_avg_pool2d(support_images, 1).squeeze_(-1).squeeze_(-1)
            query_images = F.adaptive_avg_pool2d(query_images, 1).squeeze_(-1).squeeze_(-1)

        assert support_images.dim() == query_images.dim() == 2

        support_images = F.normalize(support_images, p=2, dim=1, eps=1e-12)
        query_images = F.normalize(query_images, p=2, dim=1, eps=1e-12)
    
        X_sup = support_images.cpu().detach().numpy()
        X_query = query_images.cpu().detach().numpy()
        support_labels = support_labels.cpu().detach().numpy()

        classifier = LogisticRegression(random_state=0, max_iter=1000).fit(X=X_sup, y=support_labels)
        classification_scores = torch.from_numpy(classifier.predict_proba(X_query)).to(query_images.device)
        return classification_scores

def create_model():
    return Logistic_Regression()