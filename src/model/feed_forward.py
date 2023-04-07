from typing import Tuple

import numpy as np
import torch
from torch import nn
from src.model.neural_network import AZNeuralNetwork


class AZFeedForward(AZNeuralNetwork):
    """
    As the position of a Dots-and-Boxes game is represented as a vector (and is not representable as an image), the use
    of convolutional layers does not make sense (as opposed to the original paper). Therefore, the model makes use of
    simple fully connected layers. Furthermore, the model is initialized to random weights.
    """

    def __init__(self, game_size: int, inference_device: torch.device, model_parameters: dict):
        super(AZFeedForward, self).__init__(game_size, inference_device)

        n_lines = 2 * self.game_size * (self.game_size + 1)
        n_boxes = 2 * self.game_size

        # use parameter information from config
        hidden_layers = model_parameters["hidden_layers"]

        # hidden layers
        self.fully_connected_layers = []
        for i in range(len(hidden_layers)):
            fc_layer = nn.Sequential(
                    nn.Linear(
                        in_features=(n_lines + n_boxes if i == 0 else hidden_layers[i - 1]),
                        out_features=hidden_layers[i]
                    ),
                    nn.BatchNorm1d(hidden_layers[i]),
                    nn.ReLU(),
            )
            self.fully_connected_layers.append(fc_layer)

        self.fully_connected_layers = nn.ModuleList(self.fully_connected_layers)

        # output layers
        self.p_head = nn.Sequential(
            nn.Linear(hidden_layers[-1], n_lines),
            nn.Softmax(dim=1),
        )

        self.v_head = nn.Sequential(
            nn.Linear(hidden_layers[-1], 1),
            nn.Tanh(),
        )

        # initialize weights
        self.weight_init()


    def weight_init(self):
        """initialize fully connected layers and both output heads"""
        for layer in self.fully_connected_layers + [self.p_head, self.v_head]:
            for linear in [m for m in layer if isinstance(m, nn.Linear)]:
                nn.init.xavier_normal_(linear.weight)  # weight init
                linear.bias.data.fill_(0.01)  # bias init


    @staticmethod
    def encode(l: np.ndarray, b: np.ndarray) -> np.ndarray:
        """no encoding needed"""
        x = np.concatenate([l, b.flatten()], dtype=np.float32)
        return x


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        for layer in self.fully_connected_layers:
            x = layer(x)

        p = self.p_head(x)
        v = self.v_head(x).squeeze()  # one-dimensional output

        return p, v
