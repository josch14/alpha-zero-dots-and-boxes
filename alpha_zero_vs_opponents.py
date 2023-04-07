import argparse
import os
import torch

# local import
from src import Evaluator, AZDualRes, AZFeedForward, AlphaBetaPlayer, NeuralNetworkPlayer, RandomPlayer, Checkpoint


"""
Example call:
python alpha_zero_vs_opponents.py -cp alpha_zero_2x2 -w 4 -n 500
python alpha_zero_vs_opponents.py -cp alpha_zero_3x3 -w 4 -n 500
python alpha_zero_vs_opponents.py -cp alpha_zero_4x4 -w 4 -n 500
"""

parser = argparse.ArgumentParser()
parser.add_argument('-cp', '--checkpoint', type=str, required=True,
                    help='Model checkpoint (i.e., name of folder containing config and model).')
parser.add_argument('-w', '--n_workers', type=int, default=4,
                    help='Number of threads during self-play. Each thread performs games of self-play.')
parser.add_argument('-n', '--n_games', type=int, default=1000,
                    help='Number of games played against each opponent.')
parser.add_argument('-idev', '--inference_device', type=str, default="cpu", choices=["cpu", "cuda"],
                    help='Device with which model interference is performed during MCTS.')
args = parser.parse_args()


class Helper:

    def __init__(self, config: dict, n_workers: int, inference_device: str, checkpoint: Checkpoint, n_games: int):

        self.checkpoint = checkpoint
        self.n_workers = n_workers
        self.n_games = n_games

        self.game_size = config["game_size"]
        self.mcts_parameters = config["mcts_parameters"]
        self.model_parameters = config["model_parameters"]

        # utilize gpu if desired
        if "cuda" in inference_device:
            assert torch.cuda.is_available()
        self.inference_device = torch.device(inference_device)
        print(f"\nModel inference device: {self.inference_device}")

        # initialize model
        AZModel = None
        if self.model_parameters["name"] == "FeedForward":
            AZModel = AZFeedForward
            pass
        elif self.model_parameters["name"] == "DualRes":
            AZModel = AZDualRes

        self.model = AZModel(
            game_size=self.game_size,
            inference_device=self.inference_device,
            model_parameters=self.model_parameters,
        ).float()


    def play_games(self):

        self.model.load_checkpoint(self.checkpoint.model)

        # evaluator: model comparison against non-neural network players
        print("\n-------------- Model Comparison --------------")
        neural_network_player = NeuralNetworkPlayer(
            model=self.model,
            name=f"AlphaZero({self.game_size}x{self.game_size})",
            mcts_parameters=self.mcts_parameters,
            device=self.inference_device
        )

        for opponent in [RandomPlayer(), AlphaBetaPlayer(depth=1), AlphaBetaPlayer(depth=2), AlphaBetaPlayer(depth=3)]:
            Evaluator(
                game_size=self.game_size,
                player1=neural_network_player,
                player2=opponent,
                n_games=self.n_games,
                n_workers=self.n_workers
            ).compare()


if __name__ == '__main__':

    LOGS_FOLDER = "logs/"

    checkpoint_folder = LOGS_FOLDER + args.checkpoint + "/"
    if not os.path.exists(checkpoint_folder):
        exit(f"loading Checkpoint failed: {checkpoint_folder} does not exist")

    # create checkpoint handler
    checkpoint = Checkpoint(checkpoint_folder)

    # load config from checkpoint
    config = checkpoint.load_config()


    helper = Helper(
        config=config,
        n_workers=args.n_workers,
        inference_device=args.inference_device,
        checkpoint=checkpoint,
        n_games=args.n_games
    )
    helper.play_games()
