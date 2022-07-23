import torch
import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, ip_size, hid_size, no_of_classes):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(ip_size, hid_size)
        self.layer2 = nn.Linear(hid_size, hid_size)
        self.layer3 = nn.Linear(hid_size, no_of_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)

        return out