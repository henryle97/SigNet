from torch import nn
import torch


class SigNet(nn.Module):
    def __init__(self):
        super().__init__()

        # input_size: [155,220,1]  ~ height, width, channel
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=1),
            nn.ReLU(),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(96, 256, 5, stride=1, padding=2, padding_mode='zeros'),
            nn.ReLU(),
            nn.LocalResponseNorm(alpha=1e-4, beta=0.75, k=2, size=5),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),

            nn.Conv2d(256, 384, 3, stride=1, padding=1,  padding_mode='zeros'),
            nn.Conv2d(384, 256, 3, stride=1, padding=1,  padding_mode='zeros'),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(p=0.3),

            nn.Flatten(1, -1),
            nn.Linear(18*26*256, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, 128)
        )

    def forward(self, input_1, input_2):
        output_1 = self.feature_extractor(input_1)
        output_2 = self.feature_extractor(input_2)

        return output_1, output_2


def load_model_from_checkpoint(checkpoint_path):
    model = SigNet()
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    # torch.save(model.state_dict(), "checkpoint_weight.pth")

    return model

def load_model_from_weight(weight_path):
    model = SigNet()
    model.load_state_dict(torch.load(weight_path))

    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = SigNet()
    print(model)
    # [batch_size, channel, height, width]
    x_1 = torch.rand((1, 1, 155, 220))
    x_2 = torch.rand((1, 1, 155, 220))

    out1, out2 = model(x_1, x_2)
    print(out1.shape)
    print(count_parameters(model))




