import torch.nn as nn
import torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, in_channels=4, num_actions=6):
        super(QNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc4 = nn.Linear(7 * 7 * 64, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.fc4(x.view(x.size(0), -1)))
        return self.fc5(x)


class QNetLinear(nn.Module):
    def __init__(self, in_channels=4, num_actions=6):
        super(QNetLinear, self).__init__()
        self.fc = nn.Linear(84*84*4, num_actions)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
