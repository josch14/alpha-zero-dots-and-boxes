import json
import matplotlib.pyplot as plt
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, required=True,
                    help='Directory containing results of AlphaZero training.')
parser.add_argument('-s', '--game_size', type=int, required=True,
                    help='Size of the Dots and Boxes game for which the model was trained.')
parser.add_argument('-n', '--n_iterations', type=int, default=None,
                    help='Limit for number of iterations to show.')
args = parser.parse_args()


def plot(data: dict, n_iterations: int, game_size: int):

    fig, axs = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.tight_layout()
    title = f"{game_size}x{game_size} Dots and Boxes: AlphaZero vs. Opponents"
    fig.suptitle(title, fontsize=14, weight='bold')


    for i, statistic_type in enumerate(["starting", "second", "total"]):

        # single plot type: iterate over opponents and plot results
        for opponent, results_per_iteration in data.items():

            if opponent == "RandomPlayer":
                continue
            elif opponent == "AlphaBetaPlayer(Depth=1)":
                color = "darkorange"
            elif opponent == "AlphaBetaPlayer(Depth=2)":
                color = "red"
            elif opponent == "AlphaBetaPlayer(Depth=3)":
                color = "darkred"

            results_per_iteration = results_per_iteration if n_iterations is None \
                else results_per_iteration[:n_iterations]
            results = [e[statistic_type] for e in results_per_iteration]
            n_games = sum(results[0])
            win_percents = calculate_win_percents(results)

            axs[i].plot(
                list(range(1, len(win_percents) + 1)),
                win_percents,
                label=opponent,
                marker=".",
                markersize=8,
                linewidth=1.5,
                color=color
            )

        subplot_title = None
        # label subplot
        if statistic_type == "starting":
            subplot_title = f"First Move: AlphaZero ({n_games} games)"
        elif statistic_type == "second":
            subplot_title = f"First Move: Opponent ({n_games} games)"
        else:
            subplot_title = f"First Move: 50% AlphaZero, 50% Opp. ({n_games} games)"
        axs[i].set_title(subplot_title, fontsize=12)

        # x-axis
        axs[i].set_xlabel("Training Iterations")
        # axs[i].set_xlim(0.95, len(win_percents)+0.05)
        axs[i].set_xticks(list(range(5, len(win_percents) + 1, 5)))  # Set text labels and properties.

        # y-axis
        axs[i].set_ylabel("Alpha Zero: Win %")
        axs[i].set_ylim(-0.02, 1.02)
        axs[i].set_yticks([0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
                          ["0%", "20%", "40%", "50%", "60%", "80%", "100%"])

        # legend and grid
        legend = axs[i].legend(fontsize=9)
        legend.set_title('Opponent', prop={'size': 10})
        axs[i].grid()

    plt.subplots_adjust(left=0.05,
                        bottom=0.1,
                        right=0.98,
                        top=0.86,
                        wspace=0.255,
                        hspace=0.0)
    plt.show()
    # fig.savefig(f"./images/loss_and_results/iteration_game_results_{args.game_size}x{args.game_size}.svg", format='svg', dpi=1200)

def calculate_win_percents(results):
    return [(wins + 0.5 * draws)/(wins + draws + losses) for wins, draws, losses in results]


if __name__ == '__main__':

    filename = args.directory + './results.json'
    with open(filename) as f:
        data = json.load(f)

    plot(data, args.n_iterations, args.game_size)
