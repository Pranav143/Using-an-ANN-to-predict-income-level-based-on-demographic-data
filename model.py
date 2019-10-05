import torch.nn as nn
import torch
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_size):
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, features):
        features = torch.sigmoid(self.fc2(F.relu(self.fc1(features))))
        return features


