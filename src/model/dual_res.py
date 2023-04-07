import logging
from typing import Tuple

import numpy as np
import torch
from torch import nn

from src import DotsAndBoxesGame
from src.model.neural_network import AZNeuralNetwork


class AZDualRes(AZNeuralNetwork):

    def __init__(self, game_size: int, inference_device: torch.device, model_parameters: dict):
        super(AZDualRes, self).__init__(game_size, inference_device)

        img_size = 2 * game_size + 1

        # use parameter information from config
        blocks_params = model_parameters["blocks"]
        residual_blocks = blocks_params["residual_blocks"]
        channels = blocks_params["channels"]
        conv_kernel_size = blocks_params["conv_kernel_size"]
        res_kernel_size = blocks_params["res_kernel_size"]
        stride = blocks_params["stride"]
        padding = blocks_params["padding"]

        heads_params = model_parameters["heads"]
        policy_head_channels = heads_params["policy_head_channels"]
        value_head_channels = heads_params["value_head_channels"]
        heads_kernel_size = heads_params["heads_kernel_size"]
        heads_stride = heads_params["heads_stride"]
        heads_padding = heads_params["heads_padding"]

        # convolutional block
        self.conv_block = ConvBlock(
            out_channels=channels,
            kernel_size=conv_kernel_size,
            stride=stride,
            padding=padding
        )

        # residual blocks
        self.residual_blocks = nn.ModuleList(
            [ResBlock(
                n_channels=channels,
                kernel_size=res_kernel_size,
                stride=stride,
                padding=padding
            ) for _ in range(residual_blocks)]
        )

        # policy head
        self.policy_head = PolicyHead(
            conv_in_channels=channels,
            conv_out_channels=policy_head_channels,
            kernel_size=heads_kernel_size,
            stride=heads_stride,
            padding=heads_padding,
            fc_in_features=(policy_head_channels * img_size * img_size),
            fc_out_features=(2 * self.game_size * (self.game_size + 1))  # dimension of policy vector
        )

        # value head (dimension=1 for resulting value)
        self.value_head = ValueHead(
            conv_in_channels=channels,
            conv_out_channels=value_head_channels,
            kernel_size=heads_kernel_size,
            stride=heads_stride,
            padding=heads_padding,
            fc_in_features=(value_head_channels * img_size * img_size)
        )

        # initialize weights
        self.weight_init()
        self.float()


    def weight_init(self):
        """initialize model weights"""

        # conv block
        conv2d = self.conv_block.conv
        nn.init.xavier_normal_(conv2d.weight)
        conv2d.bias.data.fill_(0.01)

        # residual blocks
        for res_block in self.residual_blocks:
            # conv1
            nn.init.xavier_normal_(res_block.conv1.weight)
            res_block.conv1.bias.data.fill_(0.01)
            # conv2
            nn.init.xavier_normal_(res_block.conv2.weight)
            res_block.conv2.bias.data.fill_(0.01)

        # policy head
        nn.init.xavier_normal_(self.policy_head.conv.weight)
        self.policy_head.conv.bias.data.fill_(0.01)
        # fc
        nn.init.xavier_normal_(self.policy_head.fc.weight)
        self.policy_head.fc.bias.data.fill_(0.01)

        # value head
        nn.init.xavier_normal_(self.value_head.conv.weight)
        self.value_head.conv.bias.data.fill_(0.01)
        # fc1
        nn.init.xavier_normal_(self.value_head.fc1.weight)
        self.value_head.fc1.bias.data.fill_(0.01)
        # fc2
        nn.init.xavier_normal_(self.value_head.fc2.weight)
        self.value_head.fc2.bias.data.fill_(0.01)

    @staticmethod
    def encode(l: np.ndarray, b: np.ndarray) -> np.ndarray:
        """encode lines and boxes into images"""

        game_size = DotsAndBoxesGame.n_lines_to_size(l.size)
        img_size = 2 * game_size + 1

        img_l = np.zeros((img_size, img_size), dtype=np.float32)
        img_b_player = np.zeros((img_size, img_size), dtype=np.float32)
        img_b_opponent = np.zeros((img_size, img_size), dtype=np.float32)
        img_background = np.zeros((img_size, img_size), dtype=np.float32)

        # 1) image containing information which lines are drawn (for policy prediction)
        h, v = DotsAndBoxesGame.l_to_h_v(l)
        # horizontals: even rows, odd columns (0-indexing)
        # verticals: odd rows, even columns (0-indexing)
        img_l[::2, 1::2] = h
        img_l[1::2, ::2] = v
        img_l[img_l == -1.0] = 1.0

        # 2) image indicating boxes captured by player (for value prediction)
        img_b_player[1::2, 1::2] = b
        img_b_player[img_b_player == -1.0] = 0.0

        # 3) image indicating boxes captured by opponent (for value prediction)
        img_b_opponent[1::2, 1::2] = b
        img_b_opponent[img_b_opponent == 1.0] = 0.0
        img_b_opponent[img_b_opponent == -1.0] = 1.0

        # 4) image indicating unimportant pixels
        img_background[0::2, 0::2] = np.ones((game_size+1, game_size+1), dtype=np.float32)

        feature_planes = np.stack([img_l, img_b_player, img_b_opponent, img_background], axis=0)
        return feature_planes


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.conv_block(x)
        for res_block in self.residual_blocks:
            x = res_block(x)

        p = self.policy_head(x)
        v = self.value_head(x).squeeze()  # one-dimensional output

        return p, v


class ConvBlock(nn.Module):

    def __init__(self, out_channels, kernel_size, stride, padding):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(4, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))

        return x


class ResBlock(nn.Module):

    def __init__(self, n_channels, kernel_size, stride, padding):
        super(ResBlock, self).__init__()

        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size, stride, padding)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size, stride, padding)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        x_in = x
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += x_in
        x = self.relu2(x)

        return x


class PolicyHead(nn.Module):

    def __init__(self, conv_in_channels, conv_out_channels, kernel_size, stride, padding, fc_in_features, fc_out_features):
        super(PolicyHead, self).__init__()

        self.conv = nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(conv_out_channels)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(
            in_features=fc_in_features,
            out_features=fc_out_features
        )
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, x):

        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc(x)
        x = self.log_softmax(x).exp()

        return x


class ValueHead(nn.Module):

    def __init__(self, conv_in_channels, conv_out_channels, kernel_size, stride, padding, fc_in_features):
        super(ValueHead, self).__init__()

        self.conv = nn.Conv2d(conv_in_channels, conv_out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(conv_out_channels)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.fc1 = nn.Linear(
            in_features=fc_in_features,
            out_features=(fc_in_features//2)
        )
        self.fc2 = nn.Linear(
            in_features=(fc_in_features//2),
            out_features=1
        )


    def forward(self, x):

        x = self.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.relu(self.fc1(x))
        x = self.tanh(self.fc2(x))
        return x
