# local import
import torch

from .player import AIPlayer
from .. import MCTS
from ..game import DotsAndBoxesGame
from ..model.feed_forward import AZNeuralNetwork


class NeuralNetworkPlayer(AIPlayer):

    def __init__(self, model: AZNeuralNetwork, name: str, mcts_parameters: dict, device: torch.device):
        super().__init__(name=name)
        self.model = model
        self.mcts_parameters = mcts_parameters

        # model inference
        model.eval()
        model.to(device)

    def determine_move(self, s: DotsAndBoxesGame) -> int:
        move = MCTS.determine_move(self.model, s, self.mcts_parameters)
        return move
