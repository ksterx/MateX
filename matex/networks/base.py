from typing import List, Optional, Union

import torch
from torch import nn


def activation_from_str(activation: str):
    if activation == "relu":
        return nn.ReLU()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "leakyrelu":
        return nn.LeakyReLU()
    elif activation == "softmax":
        return nn.Softmax(dim=1)
    elif activation == "logsoftmax":
        return nn.LogSoftmax(dim=1)
    else:
        raise ValueError(f"Activation {activation} not supported")


class MLP(nn.Module):
    def __init__(
        self,
        layers: List[int],
        activation: str,
        output_activation: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        """Fully connected block with batch normalization
        Args:
            layers (List[int]): Description of size. [input, hidden, hidden, ..., output]
            activation (str, optional): Activation function
            softmax (bool, optional): Whether to add a softmax layer at the end
        Example:
            >>> net = FCBN(layers=[10, 100, 100, 2])
        """

        super().__init__()

        self.net = nn.Sequential()

        if len(layers) < 3:
            raise ValueError("Layers list must be of length 3 or more")

        for i in range(len(layers)):
            if i < len(layers) - 1:
                self.net.add_module(f"fc{i+1}", nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                self.net.add_module(f"{activation}{i+1}", activation_from_str(activation))
            elif i == len(layers) - 2 and output_activation is not None:
                self.net.add_module(
                    f"{output_activation}{i+1}", activation_from_str(output_activation)
                )
            elif i == len(layers) - 2:
                break
            else:
                notice.critical(i, len(layers))
                raise ValueError("Unexpected error")

        self.net.to(get_device(device))

    def forward(self, x):
        return self.net(x)


class MLPBN(nn.Module):
    def __init__(
        self,
        layers: List[int],
        activation: str,
        output_activation: Optional[str] = None,
        device: Optional[Union[str, torch.device]] = None,
        *args,
        **kwargs,
    ):
        """Fully connected block with batch normalization
        Args:
            layers (List[int]): Description of size. [input, hidden, hidden, ..., output]
            activation (str, optional): Activation function
            softmax (bool, optional): Whether to add a softmax layer at the end
        Example:
            >>> net = FCBN(layers=[10, 100, 100, 2])
        """

        super().__init__()
        self.net = nn.Sequential()
        for i in range(len(layers)):
            if i < len(layers) - 1:
                self.net.add_module(f"fc{i+1}", nn.Linear(layers[i], layers[i + 1]))
                self.net.add_module(f"bn{i+1}", nn.BatchNorm1d(layers[i + 1]))
                self.net.add_module(f"{activation}{i+1}", activation_from_str(activation))
            elif i == len(layers) - 1 and output_activation is not None:
                self.net.add_module(
                    f"{output_activation}{i+1}", activation_from_str(output_activation)
                )
            else:
                raise ValueError("Unexpected error. e.g. Layers list must be of length 3 or more")

        self.net.to(get_device(device))

    def forward(self, x):
        return self.net(x)


class QNet(nn.Module):
    def __init__(self, state_size, action_size, hidden_size, device):
        super().__init__()

        self.action_size = action_size
        self.device = device

        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

        print(self.net)

    def forward(self, x):
        return self.net(x)
