import prior_networks
import torch
import torch.nn as nn


class ModelFactory(object):
    MODEL_DICT = {'vgg11': prior_networks.models.vgg11,
                  'vgg11_bn': prior_networks.models.vgg11_bn,
                  'vgg13': prior_networks.models.vgg13,
                  'vgg13_bn': prior_networks.models.vgg13_bn,
                  'vgg16': prior_networks.models.vgg16,
                  'vgg16_bn': prior_networks.models.vgg16_bn,
                  'vgg19': prior_networks.models.vgg19,
                  'vgg19_bn': prior_networks.models.vgg19_bn,
                  'myvgg16': prior_networks.models.myvgg16,
                  'myvgg16_bn': prior_networks.models.myvgg16_bn,
                  'myvgg19': prior_networks.models.myvgg19,
                  'myvgg19_bn': prior_networks.models.myvgg19_bn,
                  'resnet18': prior_networks.models.resnet18,
                  'resnet34': prior_networks.models.resnet34,
                  'resnet50': prior_networks.models.resnet50,
                  'resnet101': prior_networks.models.resnet101,
                  'resnet152': prior_networks.models.resnet152,
                  'resnext50_32x4d': prior_networks.models.resnext50_32x4d,
                  'resnext101_32x8d': prior_networks.models.resnext101_32x8d,
                  'wide_resnet50_2': prior_networks.models.wide_resnet50_2,
                  'wide_resnet101_2': prior_networks.models.wide_resnet101_2,
                  'wide_resnet28_10': prior_networks.models.wide_resnet28_10,
                  'wide_leaky_resnet28_10': prior_networks.models.wide_leaky_resnet28_10,
                  'wide_resnet28_12': prior_networks.models.wide_resnet28_12,
                  'densenet121': prior_networks.models.densenet121,
                  'densenet161': prior_networks.models.densenet161,
                  'densenet169': prior_networks.models.densenet169,
                  'densenet201': prior_networks.models.densenet201}
    ARCHITECTURE_FIELD = 'arch'
    STATE_DICT_FIELD = 'model_state_dict'
    MODEL_ARGS_FIELDS = ['num_classes', 'small_inputs', 'dropout_rate']
    EXPECTED_FIELDS = ['num_classes', 'small_inputs', 'n_channels', 'n_in', 'dropout_rate']

    @classmethod
    def model_from_checkpoint(cls, checkpoint) -> nn.Module:
        model_class = cls.MODEL_DICT[checkpoint[cls.ARCHITECTURE_FIELD]]

        model_param_dict = {field: checkpoint[field] for field in cls.MODEL_ARGS_FIELDS}
        model = model_class(**model_param_dict)

        model.load_state_dict(checkpoint[cls.STATE_DICT_FIELD])
        return model

    @classmethod
    def checkpoint_model(cls, path, model, arch, **kwargs):
        assert arch in cls.MODEL_DICT.keys()
        assert set(kwargs.keys()) == set(cls.EXPECTED_FIELDS)

        checkpoint_dict = kwargs
        checkpoint_dict[cls.STATE_DICT_FIELD] = model.state_dict()
        checkpoint_dict[cls.ARCHITECTURE_FIELD] = arch
        torch.save(checkpoint_dict, path)

    @classmethod
    def create_model(cls, arch, **kwargs):
        assert arch in cls.MODEL_DICT.keys()

        model = cls.MODEL_DICT[arch](**kwargs)
        return model
