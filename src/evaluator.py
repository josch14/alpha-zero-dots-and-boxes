from multiprocessing import Pool
from sys import stdout
from typing import Tuple
from tqdm import tqdm

# local import
from .game import DotsAndBoxesGame
from .players.player import AIPlayer


class Evaluator:
    """
    Let two players (AIs) play games of Dots-and-Boxes against each other.

    Attributes
    ----------
    game_size : int
        board size (width & height) of a Dots-and-Boxes game
    player1, player2 : AIPlayer, AIPlayer
        AI players that are compared against each other
    n_games : int
        number of games the models play against each other (50% as starting player, 50% as second player)
    """

    def __init__(self, game_size: int, player1: AIPlayer, player2: AIPlayer, n_games: int, n_workers: int):

        self.game_size = game_size
        self.player1 = player1
        self.player2 = player2
        self.n_games = n_games
        self.n_workers = n_workers

    def compare(self) -> Tuple[int, int, int]:

        print(f"Comparing {self.player1.name}:Draw:{self.player2.name} ... ")
        results = {"starting": [0, 0, 0], "second": [0, 0, 0], "total": [0, 0, 0]}

        for i, starting in enumerate([True, False]):
            key = "starting" if starting else "second"
            if self.n_workers > 1:
                with Pool(processes=self.n_workers) as pool:

                    for result in pool.istarmap(self.play_game, tqdm([(starting,)] * (self.n_games // 2), file=stdout, smoothing=0.0)):

                        if result == 1:
                            results[key][0] += 1
                        elif result == -1:
                            results[key][2] += 1
                        else:
                            results[key][1] += 1
            else:
                for _ in tqdm(range(self.n_games // 2), file=stdout):
                    result = self.play_game(starting)

                    if result == 1:
                        results[key][0] += 1
                    elif result == -1:
                        results[key][2] += 1
                    else:
                        results[key][1] += 1

        results["total"] = [sum(x) for x in zip(results["starting"], results["second"])]

        for key in results.keys():
            print(f"Result: {results[key][0]}:{results[key][1]}:{results[key][2]} ({key})")

        return results

    def play_game(self, player1_moves_first: bool) -> int:
        game = DotsAndBoxesGame(self.game_size, (1 if player1_moves_first else -1))

        while game.is_running():
            move = self.player1.determine_move(game) if game.current_player == 1 else self.player2.determine_move(game)
            game.execute_move(move)

        return game.result
