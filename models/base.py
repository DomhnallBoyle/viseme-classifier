import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, num_classes, **kwargs):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.kwargs = kwargs
        self.model = self.create_model()
        # print(self)

    def freeze_model_params(self, model, params=None):
        """Ensure model params not trained (frozen). True by default"""
        # if you don't supply specific layers to freeze, freeze all

        if not params:
            params = model.parameters()

        for param in params:
            param.requires_grad = False

        return model

    @property
    def input_size(self):
        raise NotImplementedError

    def create_model(self):
        raise NotImplementedError

    def forward(self, images):
        raise NotImplementedError
