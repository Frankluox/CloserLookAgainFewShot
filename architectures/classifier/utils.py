import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils.weight_norm import WeightNorm


def weight_norm(module, name='weight', dim=0):
    r"""Applies weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    WeightNorm.apply(module, name, dim)
    return module

def compute_prototypes(features_support, labels_support):
    #[n_way,num_support]
    labels_support_transposed = labels_support.transpose(0, 1)

    #[n_way, dim]
    prototypes = torch.mm(labels_support_transposed, features_support)
    labels_support_transposed = (labels_support_transposed.sum(dim=1, keepdim=True)+1e-12).expand_as(prototypes)
    prototypes = prototypes.div(
        labels_support_transposed
    )
    #[n_way,dim]
    return prototypes

def prototype_scores(support_embeddings, support_labels,
                   query_embeddings):
    one_hot_label = F.one_hot(support_labels,num_classes = torch.max(support_labels).item()+1).float()
    support_embeddings = F.normalize(support_embeddings, p=2, dim=1, eps=1e-12)
    # [n_way,dim]
    prots = compute_prototypes(support_embeddings, one_hot_label)
    prots = F.normalize(prots, p=2, dim=1, eps=1e-12)

    query_embeddings = F.normalize(query_embeddings, p=2, dim=1, eps=1e-12)
    classification_scores = torch.mm(query_embeddings, prots.transpose(0, 1))*10

    return classification_scores

class Linear_normalized(nn.Linear):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(in_features, out_features, False)
    def forward(self, input):
        return F.linear(input, F.normalize(self.weight,dim=1), self.bias)


class CC_head(nn.Module):
    def __init__(self, indim, outdim, scale_cls=10.0, learn_scale=True):
        super().__init__()
        self.L = weight_norm(Linear_normalized(indim, outdim), name='weight', dim=0)
        self.scale_cls = nn.Parameter(
            torch.FloatTensor(1).fill_(scale_cls), requires_grad=learn_scale
        )

    def forward(self, features):
        assert features.dim() == 2
        x_normalized = F.normalize(features, p=2, dim=1, eps = 1e-12)
        cos_dist = self.L(x_normalized)
        classification_scores = self.scale_cls * cos_dist
        return classification_scores


def get_init_prefix(model, support_images, labels_support):
    # for eTT. Initialize trainable prefix.
    with torch.no_grad():
        one_hot_label = F.one_hot(labels_support,num_classes = torch.max(labels_support).item()+1).float()
        patch_embed = model.get_avg_patch_embed(support_images)#[b,c]
        patch_embed = F.normalize(patch_embed, p=2, dim=1, eps=1e-12)
        prefix = compute_prototypes(patch_embed, one_hot_label)#[way,c]
        prefix = F.normalize(prefix, p=2, dim=1, eps=1e-12)
    return prefix