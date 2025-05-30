import torch.nn as nn

class VerySimpleNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(VerySimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        return x