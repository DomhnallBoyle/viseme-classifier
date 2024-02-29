import torchvision.models as models

from models.base import *


class Inception(CNN):

    def __init__(self, num_classes, **kwargs):
        super(Inception, self).__init__(num_classes, **kwargs)

    @property
    def input_size(self):
        return 299

    def create_model(self):
        # by default, all params here are marked as requires_grad=True
        # i.e. they are not frozen which means entire network is trained
        model = models.inception_v3(pretrained=True, aux_logits=False)

        if self.kwargs['freeze_layers']:
            # freezing layers makes the model train faster
            model = self.freeze_model_params(
                model,
                [
                    param for name, param in model.named_parameters()
                    if name.split('.')[0] in [
                        'Conv2d_1a_3x3',
                        'Conv2d_2a_3x3',
                        'Conv2d_2b_3x3',
                        'Conv2d_3b_1x1',
                        'Conv2d_4a_3x3',
                    ]
                ]
            )

        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, model.fc.out_features),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(model.fc.out_features, 250),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(250, self.num_classes),
        )  # these are not frozen/trainable (requires_grad=True) by default

        # debug information
        for name, param in model.named_parameters():
            print(name, param.requires_grad)

        return model

    def forward(self, images):
        return self.model(images)
