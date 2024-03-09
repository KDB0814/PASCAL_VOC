import torch.nn as nn
import torchvision.models as models

class Vgg16_backbone(nn.Module):
    def __init__(self):
        super(Vgg16_backbone, self).__init__()
        self.layers = [0,5,10,17,24,31]
        self.weights = models.VGG16_Weights.DEFAULT
        self.vgg16 = models.vgg16(weights=self.weights)
        self.features = self.vgg16.features

    def forward(self, x):
        output = {}
        for i in range(len(self.layers)-1):
            for layer in range(self.layers[i], self.layers[i+1]):
                x = self.features[layer](x)
            output[f'x{i+1}'] = x
        return output
    
class O_FCN8s(nn.Module):
    def __init__(self, n_class, backbone=Vgg16_backbone()):
        super().__init__()
        self.n_class = n_class
        self.backbone = backbone
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, 3, 2, 1, dilation=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, 2, 1, dilation=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, dilation=1, output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, dilation=1, output_padding=1)
        self.deconv5 = nn.ConvTranspose2d(64, 32, 3, 2, 1, dilation=1, output_padding=1)
        self.classifier = nn.Conv2d(32, self.n_class, kernel_size=1)

    def forward(self, x):
        out_backbone = self.backbone(x)
        
        pool5 = out_backbone['x5']
        pool4 = out_backbone['x4']
        pool3 = out_backbone['x3']

        segmap = self.relu(self.deconv1(pool5))
        segmap = segmap + pool4
        segmap = self.relu(self.deconv2(segmap))
        segmap = segmap + pool3
        segmap = self.relu(self.deconv3(segmap))
        segmap = self.relu(self.deconv4(segmap))
        segmap = self.relu(self.deconv5(segmap))
        segmap = self.classifier(segmap)

        return segmap