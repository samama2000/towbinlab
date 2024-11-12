import torch
import torch.nn as nn
import torchvision.models as models

class CSRNet(nn.Module):
    def __init__(self, transform_dic=None, load_weights=False, pretrained=False):
        super(CSRNet, self).__init__()

        if pretrained:
            # Frontend (VGG-16 layers)
            vgg16 = models.vgg16_bn(pretrained = pretrained)
            self.frontend = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
                *list(vgg16.features.children())[1:33]
            )

            for param in self.frontend.parameters():
                param.requires_grad = False
        else:
            # We change the input channel in the first conv layer to 1 for grayscale images
            self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
            self.frontend = make_layers(self.frontend_feat, in_channels=1)  # Change in_channels to 1

        # Backend (Dilated convolutional layers)
        self.backend_feat = [512, 512, 512, 256, 128, 64]
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)

        # Output layer
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        
        # Add an upsample layer to resize output to match target size
        self.upsample = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=False)

        self.transform_dic = transform_dic

        if load_weights and not pretrained:
            self._initialize_weights()


    def forward(self, x):
        x = self.frontend(x)   # Process grayscale image
        x = self.backend(x)     # Backend layers
        x = self.output_layer(x)  # Final output layer
        x = self.upsample(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

# Utility function to create layers from the configuration
def make_layers(cfg, in_channels=3, dilation=False):
    """
    Function to create the layers from the given configuration.
    Now accepts 'in_channels' as an argument, which we can set to 1 for grayscale images.
    """
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if dilation:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=2, dilation=2), 
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1), 
                           nn.BatchNorm2d(v),
                           nn.ReLU(inplace=True)]
            in_channels = v  # Update the in_channels for the next layer
    return nn.Sequential(*layers)
