import argparse
import os
import time

# local import
from src import DotsAndBoxesPrinter, AlphaBetaPlayer, AIPlayer, RandomPlayer, Checkpoint
from src.model.dual_res import AZDualRes
from src.model.feed_forward import AZFeedForward
from src.players.neural_network import NeuralNetworkPlayer


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--size', type=int, default=2,
                    help='Size of the Dots-and-Boxes game (in number of boxes per row and column).')
parser.add_argument('-o', '--opponent', type=str, default="alpha_beta", choices=["person", "random", "alpha_beta", "alpha_zero"],
                    help='Type of opponent to play against (in case of AlphaZero, checkpoint needs to be passed too).')
parser.add_argument('-cp', '--checkpoint', type=str,
                    help='In case of AlphaZero as opponent: Model checkpoint (i.e., name of folder containing config and model).')
parser.add_argument('-d', '--depth', type=int, default=3,
                    help='In case of Alpha–beta pruning as opponent: Specifies the search depth')
args = parser.parse_args()


def cls(): os.system("cls" if os.name == "nt" else "clear")


def main(size: int, opponent: AIPlayer):
    cls()

    opponent_name = "Opponent" if opponent is None else opponent.name
    game = DotsAndBoxesPrinter(size, opponent_name)

    print(game.state_string())
    print(game.board_string())

    while game.is_running():

        if game.current_player == 1 or opponent is None:
            # print draw request
            print("Please enter a free line number: ", end="")

            # process draw request
            while True:
                move = int(input())
                if move in game.get_valid_moves():
                    break
                print(f"Line {move} is not a valid move. Please select a move in {game.get_valid_moves()}.")
            last_move_by_player = True

        else:
            # an AI opponent is at turn
            time.sleep(1.0)
            start_time = time.time()
            move = opponent.determine_move(game)
            stopped_time = time.time() - start_time
            last_move_by_player = False

        game.execute_move(move)

        # print new game state
        cls()
        if not last_move_by_player:
            print("Computation time of opponent for previous move {0:.2f}s".format(stopped_time))
        else:
            print()
        print(game.state_string())
        print(game.board_string())


    if game.result == 1:
        print("The game is over.. You won!")
    elif game.result == -1:
        print("The game is over.. You lost :(")
    else:
        print("The game ended in a draw ..")
    print(game.state_string())



if __name__ == '__main__':

    game_size = args.size

    if args.opponent == "person":
        opponent = None

    elif args.opponent == "random":
        opponent = RandomPlayer()

    elif args.opponent == "alpha_beta":
        opponent = AlphaBetaPlayer(depth=args.depth)

    elif args.opponent == "alpha_zero":
        LOGS_FOLDER = "logs/"

        checkpoint_folder = LOGS_FOLDER + args.checkpoint + "/"
        if not os.path.exists(checkpoint_folder):
            exit(f"loading Checkpoint failed: {checkpoint_folder} does not exist")

        # create checkpoint handler and load config
        checkpoint = Checkpoint(checkpoint_folder)
        config = checkpoint.load_config()
        inference_device = "cpu"
        assert config["game_size"] == args.size

        # initialize model
        AZModel = None
        if config["model_parameters"]["name"] == "FeedForward":
            AZModel = AZFeedForward
        elif config["model_parameters"]["name"] == "DualRes":
            AZModel = AZDualRes

        model = AZModel(
            game_size=game_size,
            inference_device="cpu",
            model_parameters=config["model_parameters"],
        ).float()
        model.load_checkpoint(checkpoint.model)

        opponent = NeuralNetworkPlayer(
            model=model,
            name=f"AlphaZero({game_size}x{game_size})",
            mcts_parameters=config["mcts_parameters"],
            device="cpu"
        )



    main(game_size, opponent)
