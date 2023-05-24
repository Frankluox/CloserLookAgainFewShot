from importlib import import_module

def get_module(module_name, config):
    model_module = import_module("." + module_name, package="modules")
    model_module = getattr(model_module, "get_model")
    return model_module(config)

def build_model(config):
    model = get_module(config.MODEL.TYPE, config)
    return model