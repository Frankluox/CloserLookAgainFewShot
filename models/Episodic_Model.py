"""
Module for episodic (meta) training/testing
"""
from architectures import get_backbone, get_classifier
import torch.nn as nn
import torch.nn.functional as F
from utils import accuracy

class EpisodicTraining(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.backbone = get_backbone(config.MODEL.BACKBONE, *config.MODEL.BACKBONE_HYPERPARAMETERS)
        self.classifier = get_classifier(config.MODEL.CLASSIFIER, *config.MODEL.CLASSIFIER_PARAMETERS)
    
    def forward(self,img_tasks,label_tasks, *args, **kwargs):
        batch_size = len(img_tasks)
        loss = 0.
        acc = []
        for i, img_task in enumerate(img_tasks):
            support_features = self.backbone(img_task["support"].squeeze_().cuda())
            query_features = self.backbone(img_task["query"].squeeze_().cuda())
            score = self.classifier(query_features, support_features,
                                    label_tasks[i]["support"].squeeze_().cuda(), **kwargs)
            loss += F.cross_entropy(score, label_tasks[i]["query"].squeeze_().cuda())
            acc.append(accuracy(score, label_tasks[i]["query"].cuda())[0])
        loss /= batch_size
        return loss, acc
    
    def train_forward(self, img_tasks,label_tasks, *args, **kwargs):
        return self(img_tasks, label_tasks, *args, **kwargs)
    
    def val_forward(self, img_tasks,label_tasks, *args, **kwargs):
        return self(img_tasks, label_tasks, *args, **kwargs)
    
    def test_forward(self, img_tasks,label_tasks, *args, **kwargs):
        return self(img_tasks, label_tasks, *args, **kwargs)

def get_model(config):
    return EpisodicTraining(config)