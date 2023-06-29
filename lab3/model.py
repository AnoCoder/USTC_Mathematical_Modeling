import torch
import torch.nn as nn


class DNN(nn.Module):
    def __init__(self, neurons: list, activation: str):
        super(DNN, self).__init__()
        activation = activation.lower()
        act_map = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid,
            'elu': nn.ELU,
            'leakyrelu': nn.LeakyReLU,
            'prelu': nn.PReLU,
            'softplus': nn.Softplus,
        }
        self.fc_layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        num_layers = len(neurons) - 1
        for i in range(num_layers):
            self.fc_layers.append(nn.Linear(neurons[i], neurons[i+1]))
            if i < num_layers - 1:
                self.activations.append(act_map[activation]())

    def forward(self, x):
        for i, layer in enumerate(self.fc_layers):
            x = layer(x)
            if i < len(self.activations):
                x = self.activations[i](x)
        return x

