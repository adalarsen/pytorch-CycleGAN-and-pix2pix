""" class ProgressiveGenerator(nn.Module):
    def __init__(self):
        """
            Is called when model is initialized.
            Args:
                image_channels. Number of color channels in image (3)
                num_classes: Number of classes we want to predict (10)
        """
        super().__init__()
        # Set up number of filters in each layer
        num_filters = [4, 8, 16, 32, 64, 128, 256, 512]


        # Define the convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=num_filters[0],
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels=num_filters[0],
            out_channels=num_filters[1],
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv3 = nn.Conv2d(
            in_channels=num_filters[2],
            out_channels=num_filters[3],
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv4 = nn.Conv2d(
            in_channels=num_filters[3],
            out_channels=num_filters[4],
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv5 = nn.Conv2d(
            in_channels=num_filters[4],
            out_channels=num_filters[5],
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv6 = nn.Conv2d(
            in_channels=num_filters[5],
            out_channels=num_filters[6],
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.conv7 = nn.Conv2d(
            in_channels=num_filters[6],
            out_channels=num_filters[7],
            kernel_size=3,
            stride=1,
            padding=1
        )

"""

"""        def update_network(self, num_layers):
            model = []

            if (num_layers <= 1):
                model = [conv1, nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(num_filters[0], eps=2e-05, momentum=0.4, affine=True, track_running_stats=True)]
            if(self.model <= 2):
                model += [conv2, nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Re"""LU(),
                nn.BatchNorm2d(num_filters[0], eps=2e-05, momentum=0.4, affine=True, track_running_stats=True))]
            if(self.model<=3):
                model += [conv3, nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(num_filters[0], eps=2e-05, momentum=0.4, affine=True, track_running_stats=True)]
            if (num_layers <= 4):
                model += [conv4, nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(num_filters[0], eps=2e-05, momentum=0.4, affine=True, track_running_stats=True)]
            if(self.model <= 5):
                model += [conv5, nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(num_filters[0], eps=2e-05, momentum=0.4, affine=True, track_running_stats=True)]
            if(self.model<=6):
                model += [conv6, nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(num_filters[0], eps=2e-05, momentum=0.4, affine=True, track_running_stats=True)]
            if(self.model<=7):
                model = [conv7, nn.MaxPool2d(kernel_size=2, stride=2),
                nn.ReLU(),
                nn.BatchNorm2d(num_filters[0], eps=2e-05, momentum=0.4, affine=True, track_running_stats=True)]

            self.model = nn.Sequential(*model)

        def forward(self, input):
            return self.model(input) """
