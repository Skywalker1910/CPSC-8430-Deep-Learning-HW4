import torch
import torch.nn as nn

class ACGANGenerator(nn.Module):
    def __init__(self, nz=100, n_classes=10, embed_size=100, features_g=64):
        super(ACGANGenerator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, embed_size)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz + embed_size, features_g * 16, 4, 1, 0, bias=False),
            self.Gnet(features_g * 16, features_g * 8, 4, 2, 1),  # 4x4 -> 8x8
            self.Gnet(features_g * 8, features_g * 4, 4, 2, 1),   # 8x8 -> 16x16
            self.Gnet(features_g * 4, features_g * 2, 4, 2, 1),   # 16x16 -> 32x32
            nn.ConvTranspose2d(features_g * 2, 3, 4, 2, 1, bias=False),  # 32x32 -> 64x64
            nn.Tanh()
        )

    def Gnet(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels).view(labels.size(0), -1, 1, 1)
        input = torch.cat((noise, label_input), 1)
        return self.net(input)

class ACGANDiscriminator(nn.Module):
    def __init__(self, n_classes=10, features_d=64):
        super(ACGANDiscriminator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, 1 * 64 * 64)  # Embedding size matches flattened image
        self.main = nn.Sequential(
            nn.Conv2d(4, features_d, 4, 2, 1, bias=False),  # Adjusted input channels to 4 (3 for image + 1 for label)
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d, features_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 2, features_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(features_d * 4, features_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(features_d * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # Fully connected layers
        self.adv_layer = nn.Sequential(nn.Linear(features_d * 8 * 4 * 4, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(features_d * 8 * 4 * 4, n_classes), nn.Softmax(dim=1))

    def forward(self, input, labels):
        batch_size = input.size(0)
        label_input = self.label_emb(labels).view(batch_size, 1, 64, 64)  # Adjusted to match the image dimension
        label_input = label_input.expand(-1, 1, 64, 64)  # Expand label input to match a single channel
        input = torch.cat((input, label_input), dim=1)  # Concatenate along the channel dimension
        features = self.main(input).view(batch_size, -1)
        validity = self.adv_layer(features)
        label = self.aux_layer(features)
        return validity, label
