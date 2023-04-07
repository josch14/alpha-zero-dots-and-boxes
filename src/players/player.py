from abc import ABC, abstractmethod

# local import
from ..game import DotsAndBoxesGame


class AIPlayer(ABC):

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def determine_move(self, s: DotsAndBoxesGame) -> int:
        pass
