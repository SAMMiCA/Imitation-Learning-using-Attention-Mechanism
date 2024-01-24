import torch.nn as nn
import torch

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b,c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ADAttentionModel(nn.Module):
    def __init__(self):

        super().__init__()

        self.fv_feture_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2),
            nn.ReLU(),
            nn.AvgPool2d(4),
            nn.Conv2d(32, 32, 3, 2),
            SELayer(32,4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self.bev_feture_extractor = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2),
            nn.ReLU(),
            nn.AvgPool2d(4),
            nn.Conv2d(32, 32, 3, 2),
            SELayer(32,4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )

        self.speed_feture_extractor = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 32)
        )

        self.longitudinal_head = nn.Sequential(
            nn.Linear(256+32+768, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh()
        )
        self.lateral_head = nn.Sequential(
            nn.Linear(256+32+768, 64),
            nn.ReLU(),
            nn.Linear(64, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Tanh()
        )
        

    def forward(self, fv, bev, speed):

        fv_fe = self.fv_feture_extractor(fv)
        bev_fe = self.bev_feture_extractor(bev)

        speed_fe = self.speed_feture_extractor(speed)
        fe = torch.cat([fv_fe, bev_fe, speed_fe], 1)
        pedal = self.longitudinal_head(fe)
        steer = self.lateral_head(fe)

        return pedal, steer

if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = ADAttentionModel().to(device)

    fv = torch.ones([5, 3, 150, 100]).to(device)
    bev = torch.ones([5, 3, 100, 400]).to(device)
    speed = torch.ones([5, 1]).to(device)

    pedal, steer = model(fv, bev, speed)
    print(pedal, steer)
