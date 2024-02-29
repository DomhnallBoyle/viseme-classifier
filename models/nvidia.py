from models.base import *


class Nvidia(CNN):

    def __init__(self, num_classes, **kwargs):
        super(Nvidia, self).__init__(num_classes)

    @property
    def input_size(self):
        return 70, 320

    def create_model(self):
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2),
            nn.ELU(),
            nn.Conv2d(48, 64, 3),
            nn.ELU(),
            nn.Conv2d(64, 64, 3),
            nn.Dropout(0.25)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(in_features=64 * 2 * 33, out_features=100),
            nn.ELU(),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(),
            nn.Linear(in_features=50, out_features=self.num_classes)
        )

    def forward(self, images):
        input = images.view(images.size(0), 3, 70, 320)

        output = self.conv_layers(input)
        output = output.view(output.size(0), -1)

        output = self.linear_layers(output)

        return output
