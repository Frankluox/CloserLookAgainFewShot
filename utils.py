#Adapted from Swintransformer
import collections
import os
import torch

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger, topK, step):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config,
                  'step':step}
    if topK is not None:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}_top{topK}.pth')
    else:
        save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

def delete_checkpoint(config, topK=None, epoch = None):
    if topK is not None:
        for file_ in os.listdir(config.OUTPUT):
            # delete the topK checkpoint
            if f"top{config.SAVE_TOP_K_MODEL}" in file_:
                os.remove(os.path.join(config.OUTPUT, file_))
                break
        for j in range(config.SAVE_TOP_K_MODEL-1,topK-1, -1):
            # move the checkpoints 
            for file_ in os.listdir(config.OUTPUT):
                if f"top{j}" in file_:
                    os.rename(os.path.join(config.OUTPUT, file_),
                        os.path.join(config.OUTPUT, file_).replace(f"top{j}", f"top{j+1}"))
                    break
    elif epoch is not None:
        if os.path.exists(os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth")):
            os.remove(os.path.join(config.OUTPUT, f"ckpt_epoch_{epoch}.pth"))
        
def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')

    possible_keys = ["state_dict", "model", "models"]

    flag = True
    for key in possible_keys:
        if key in checkpoint.keys():
            the_key = key
            flag = False
            break
    if flag:
        state_dict = checkpoint
    else:    
        state_dict = checkpoint[the_key]
    
    state_keys = list(state_dict.keys())
    for i, key in enumerate(state_keys):
        if "backbone" in key:
            newkey = key.replace("backbone.", "")
            state_dict[newkey] = state_dict.pop(key)
        if "classifier" in key:
            state_dict.pop(key)
    
    msg = model.backbone.load_state_dict(state_dict,strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()

def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file

def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')

    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = [0.0]*config.SAVE_TOP_K_MODEL
    step = 0
    if 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
    if 'max_accuracy' in checkpoint:
        max_accuracy = checkpoint['max_accuracy']
        logger.info(f"load max_accuracy:{max_accuracy}")
    if 'step' in checkpoint:
        step = checkpoint['step']
        logger.info(f"load step:{step}")

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy, step