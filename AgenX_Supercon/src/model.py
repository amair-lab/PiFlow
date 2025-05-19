import torch.nn as nn


class TcPredictor(nn.Module):
    def __init__(self, input_size=526, hidden=768):  # Default to 23 features based on element encoding
        super(TcPredictor, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden, bias=True)
        self.layer2 = nn.Linear(hidden, hidden, bias=True)
        self.layer3 = nn.Linear(hidden, hidden, bias=True)
        self.layer4 = nn.Linear(hidden, 1)
        self.l_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.l_relu(self.layer1(x))
        x = self.dropout(x)
        x = self.l_relu(self.layer2(x))
        x = self.dropout(x)
        x = self.l_relu(self.layer3(x))
        x = self.layer4(x)
        return x