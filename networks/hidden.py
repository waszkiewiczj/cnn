import torch.nn as nn


class Hidden(nn.Module):

    def __init__(self, input_size, hidden_sizes, output_size):
        super(Hidden, self).__init__()
        if len(hidden_sizes) == 0:
            raise ValueError('List of hidden layers sizes cannot be empty')
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fch = [
            nn.Linear(hidden_sizes[i - 1], hidden_sizes[i])
            for i in range(1, len(hidden_sizes))
        ]
        self.fc2 = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        x = self.fc1(x)
        for fc in self.fch:
            x = fc(x)
        x = self.fc2(x)
        return x
