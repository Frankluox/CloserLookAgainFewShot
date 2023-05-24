from importlib import import_module

def get_model(config, *args, **kwargs):
    model_module = import_module("." + config.MODEL.TYPE, package="models")
    model_module = getattr(model_module, "get_model")
    return model_module(config, *args, **kwargs)