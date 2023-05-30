import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torchvision.transforms as transforms

            
def build_Torch_transform(is_train, config):
    trans = []
    if is_train:
        trans.append(transforms.RandomResizedCrop(config.DATA.IMG_SIZE))

        if config.AUG.COLOR_JITTER is not None:
            trans.append(transforms.RandomApply([
                    transforms.ColorJitter(*config.AUG.COLOR_JITTER[:-1])  # BYOL
                ], p=config.AUG.COLOR_JITTER[-1]))
        
        if config.AUG.GRAY_SCALE is not None:
            trans.append(transforms.RandomGrayscale(p=config.AUG.GRAY_SCALE))
        
        if config.AUG.GAUSSIAN_BLUR is not None:
            trans.append(transforms.RandomApply([
                transforms.GaussianBlur((3, 3), (1.0, 2.0))],
                p = config.AUG.GAUSSIAN_BLUR))
        
        trans.append(transforms.RandomHorizontalFlip(p=config.AUG.FLIP))

        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=config.AUG.MEAN,
                                     std=config.AUG.STD))
        trans = transforms.Compose(trans)

    else:
        if config.AUG.TEST_CROP:
            if config.DATA.IMG_SIZE == 84:
                size = 92
            else:
                size = int((256 / 224) * config.DATA.IMG_SIZE)
            trans.append(transforms.Resize([size, size]))
            trans.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            trans.append(transforms.Resize([config.DATA.IMG_SIZE, config.DATA.IMG_SIZE]))
        trans.append(transforms.ToTensor())
        trans.append(transforms.Normalize(mean=config.AUG.MEAN,
                                     std=config.AUG.STD))
        trans = transforms.Compose(trans)
    return trans


