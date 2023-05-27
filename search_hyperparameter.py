"""
Searching hyperparameters for gradient-based adaptation algorithms
such as finetune, eTT, TSA, URL and Cosine Classifier.
"""

import argparse
from config import get_config
import os
from logger import create_logger
from data import create_torch_dataloader
from data.dataset_spec import Split
import torch
import numpy as np
import random
import json
from utils import accuracy, AverageMeter, load_pretrained
from models import get_model
import math


def setup_seed(seed):
    """
    Fix some seeds.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def parse_option():
    parser = argparse.ArgumentParser('Searching hyperparameters', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--test-batch-size', type=int, help="test batch size for single GPU")
    parser.add_argument('--output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--pretrained', type=str, help="pretrained path") 
    parser.add_argument('--tag', help='tag of experiment')
    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config

@torch.no_grad()
def testing(config, dataset, data_loader, model):
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    accs = []
    # dataset.set_epoch()
    for idx, batches in enumerate(data_loader):
        dataset_index, imgs, labels = batches

        loss, acc = model.test_forward(imgs, labels, dataset_index)
        accs.extend(acc)
        acc = torch.mean(torch.stack(acc))

        loss_meter.update(loss.item())
        acc_meter.update(acc.item())


    accs = torch.stack(accs)
    ci = (1.96*torch.std(accs)/math.sqrt(accs.shape[0])).item()
    return acc_meter.avg, loss_meter.avg, ci


def search_hyperparameter(config):
    valid_dataloader, valid_dataset = create_torch_dataloader(Split.VALID, config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")

    model = get_model(config).cuda()

    if config.MODEL.PRETRAINED:
        load_pretrained(config, model, logger)

    if hasattr(model, 'mode') and model.mode == "NCC":
        model.append_adapter()


    logger.info("Start searching for hyperparameters.")

    epoch_range = config.SEARCH_HYPERPARAMETERS.EPOCH_RANGE
    lr_backbone_range = config.SEARCH_HYPERPARAMETERS.LR_BACKBONE_RANGE
    lr_head_range = config.SEARCH_HYPERPARAMETERS.LR_HEAD_RANGE

    if lr_backbone_range is None:
        lr_backbone_range = [0]
    if lr_head_range is None:
        lr_head_range = [0]
    
    path = os.path.join(config.OUTPUT, "results.json")
    with open(path, 'w') as f:
        dic = []
        json.dump(dic, f)

    # [accuracy, confidence interval]
    max_accuracy = [0.,0.]
    logger.info(f"epoch range: {epoch_range}, backbone lr range: {lr_backbone_range}, head lr range: {lr_head_range}")
    for epoch in epoch_range:
        for lr_backbone in lr_backbone_range:
            for lr_head in lr_head_range:
                model.classifier.ft_epoch = epoch
                model.classifier.ft_lr_1 = lr_backbone
                model.classifier.ft_lr_2 = lr_head
                acc1, loss, ci = testing(config, valid_dataset, valid_dataloader, model)
                logger.info(f"Test Accuracy with epoch: {epoch}, backbone lr: {lr_backbone}, head lr: {lr_head} is {acc1:.2f}%+-{ci:.2f}")
                if acc1>max_accuracy[0]:
                    max_accuracy = [acc1, ci]
                    max_hyperparameters = (epoch, lr_backbone, lr_head)
                    logger.info("achieve new best.")

                
                with open(path, 'r') as f:
                    dic = json.load(f)
                dic.append([epoch, lr_backbone, lr_head, acc1, ci])
                with open(path, 'w') as f:
                    json.dump(dic, f)
    logger.info(f"best accuracy {max_accuracy[0]:.2f}%+-{max_accuracy[1]:.2f} is achieved when epoch is {max_hyperparameters[0]}, backbone lr is {max_hyperparameters[1]}, head lr is {max_hyperparameters[2]}.")

if __name__ == '__main__':
    args, config = parse_option()
    torch.cuda.set_device(config.GPU_ID)
    
    setup_seed(config.SEED)
    
    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, name=f"{config.MODEL.NAME}")

    path = os.path.join(config.OUTPUT, "config.json")
    with open(path, "w") as f:
        f.write(config.dump())
    logger.info(f"Full config saved to {path}")
    
    assert isinstance(config.SEARCH_HYPERPARAMETERS.EPOCH_RANGE, list)
    assert isinstance(config.SEARCH_HYPERPARAMETERS.LR_HEAD_RANGE, list) or isinstance(config.SEARCH_HYPERPARAMETERS.LR_BACKBONE_RANGE, list)
    
    search_hyperparameter(config)
