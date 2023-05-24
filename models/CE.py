"""
Simple supervised Cross Entropy (CE) training used widely in image classification.
"""
from architectures import get_backbone, get_classifier
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy
import torch
class CrossEntropyTraining(nn.Module):
    def __init__(self, config, num_classes):
        """
        config: configuration file
        num_classes: a list containing number of classes per dataset. 
                    (For CE, the number of datasets must be 1)
        """
        super().__init__()
        assert len(num_classes) == 1
        self.backbone = get_backbone(config.MODEL.BACKBONE, *config.MODEL.BACKBONE_HYPERPARAMETERS)
        self.classifier = nn.Linear(self.backbone.outdim, num_classes[0])
        self.val_test_classifier = get_classifier(config.MODEL.CLASSIFIER, *config.MODEL.CLASSIFIER_PARAMETERS)
    
    
    
    def train_forward(self, imgs,labels, *args, **kwargs):
        imgs = imgs.squeeze_().cuda()
        labels = labels.squeeze_().cuda()

        features = self.backbone(imgs)
        if features.dim() == 4:
            features = F.adaptive_avg_pool2d(features, 1).squeeze_(-1).squeeze_(-1)
        score = self.classifier(features)
        loss = F.cross_entropy(score, labels)
        acc = [accuracy(score, labels)[0]]
        return loss, acc

    def val_test_forward(self,img_tasks,label_tasks, *args, **kwargs):
        batch_size = len(img_tasks)
        loss = 0.
        acc = []
        for i, img_task in enumerate(img_tasks):
            support_features = self.backbone(img_task["support"].squeeze_().cuda())
            query_features = self.backbone(img_task["query"].squeeze_().cuda())
            score = self.val_test_classifier(query_features, support_features,
                                    label_tasks[i]["support"].squeeze_().cuda(), **kwargs)
            loss += F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda())
            acc.append(accuracy(score, label_tasks[i]["query"].cuda())[0])
        loss /= batch_size
        return loss, acc
    
    def val_forward(self, img_tasks,label_tasks, *args, **kwargs):
        return self.val_test_forward(img_tasks, label_tasks)
    
    def test_forward(self, img_tasks,label_tasks, *args,  **kwargs):
        return self.val_test_forward(img_tasks, label_tasks)

def get_model(config, num_classes):
    return CrossEntropyTraining(config, num_classes)