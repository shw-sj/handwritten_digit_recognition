import torch.nn as nn


class BPNetwork(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BPNetwork, self).__init__()

        self.model = nn.Sequential(

            nn.Linear(input_size, hidden_size),

            nn.ReLU(),

            nn.Linear(hidden_size, hidden_size),

            nn.ReLU(),

            nn.Linear(hidden_size, output_size)

        )

    def forward(self, x):

        x = self.model(x)

        return x