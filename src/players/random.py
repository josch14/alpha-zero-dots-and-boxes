import random

# local import
from .player import AIPlayer
from ..game import DotsAndBoxesGame


class RandomPlayer(AIPlayer):

    def __init__(self):
        super().__init__("RandomPlayer")

    def determine_move(self, s: DotsAndBoxesGame) -> int:
        move = random.choice(s.get_valid_moves())
        return move
