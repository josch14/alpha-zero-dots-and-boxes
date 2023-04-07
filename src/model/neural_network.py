import os
from abc import ABC, abstractmethod
from typing import Tuple
import logging

import numpy as np
import torch
from torch import nn


class AZNeuralNetwork(ABC, nn.Module):
    """
    AlphaZero neural network f(s) = (p,v) implementation, combining the roles of a policy network and value network, with
    - p (policy vector): vector of move probabilities p = P(a|s)
    - v (scalar value): probability of the current player winning from position s
    """

    def __init__(self, game_size: int, inference_device: torch.device):
        super(AZNeuralNetwork, self).__init__()

        self.inference_device = inference_device
        self.game_size = game_size

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward though the neural network. In this simple form, this method is only used during training of the
        neural network. During self-play using MCTS and model evaluation, p_v() is used which makes use of this method.

        NOTE: Input to e.g. nn.Linear is expected to be [batch_size, features].
            Therefore, single vectors have to be fed as row vectors.

        Parameters
        ----------
        x : torch.tensor
            encoding of the game state, assumed to be in its canonical form

        Returns
        -------
        p, v : [torch.Tensor, torch.Tensor]
            policy vector p (potentially containing values > 0 for invalid moves), value v
        """

        pass


    @staticmethod
    @abstractmethod
    def encode(l: np.ndarray, b: np.ndarray) -> np.ndarray:
        pass


    def p_v(self, l: np.ndarray, b: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Performs a single forward through the neural network. As opposed to forward(), this method ensures that in p
        invalid moves have probability 0 (while still ensuring a probability distribution in p).

        Parameters
        ----------
        l : np.ndarray
            lines vector, assumed to be in its canonical form
        b : np.ndarray
            boxes matrix, assumed to be in its canonical form

        Returns
        -------
        p, v : [np.ndarray, float]
            policy vector p (containing values >= 0 only for valid moves), value v
        """
        valid_moves = np.where(l == 0)[0].tolist()
        assert len(valid_moves) > 0, "No valid move left, model should not be called in this case"

        # model expects ...
        x = self.encode(l, b)
        x = torch.from_numpy(x).to(self.inference_device)  # ... tensor
        x = x.unsqueeze(0)  # ... batch due to batch normalization

        # cpu only necessary when gpu is used
        with torch.no_grad():
            p, v = self.forward(x)
        p = p.squeeze().detach().cpu().numpy()
        v = v.detach().cpu().item()

        # p possibly contains p > 0 for invalid moves -> erase those
        valid = np.zeros(l.squeeze().shape, dtype=np.float32)
        valid[valid_moves] = 1

        p_valid = np.multiply(p, valid)
        if np.sum(p_valid) == 0:
            logging.warning(f"Model did not return a probability larger than zero for any valid move:\n"
                            f"(p,v) = {(p, v)} with valid moves {valid_moves}.")
            # set probability equally for all valid moves
            p_valid = np.multiply([1] * l.shape[0], valid)

        # normalization to sum 1
        p = p_valid / np.sum(p_valid)

        return p, v


    def save_checkpoint(self, model_path: str):
        torch.save(self.state_dict(), model_path)


    def load_checkpoint(self, model_path: str):
        self.load_state_dict(torch.load(model_path))
