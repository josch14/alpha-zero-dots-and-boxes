import json
import matplotlib.pyplot as plt
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--directory', type=str, required=True,
                    help='Folder containing loss statistics of AlphaZero training.')
parser.add_argument('-s', '--game_size', type=int, required=True,
                    help='Size of the Dots and Boxes game for which the model was trained.')
parser.add_argument('-m', '--smooth', type=int, default=100,
                    help='Smoothing parameter for train loss plot (number of batches for which loss is averaged).')
parser.add_argument('-n', '--n_iterations', type=int, default=None,
                    help='Limit for number of iterations to show.')
parser.add_argument('-y', '--y_max', type=float, required=False,
                    help='Limit for number of iterations to show.')
args = parser.parse_args()


def moving_avg(x: np.ndarray, n: int) -> np.ndarray:
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[n:] - cumsum[:-n]) / float(n)


def plot(data_train_loss: dict,
         data_iteration_loss: dict,
         n_batches_smooth: int,
         n_iterations: int,
         game_size: int,
         y_max: float):

    fig, axs = plt.subplots(1, 2, figsize=(7, 4.5))
    fig.tight_layout()
    title = f"{game_size}x{game_size} Dots and Boxes: Loss Evolution"
    fig.suptitle(title, fontsize=14, weight='bold')

    data_train_loss = data_train_loss if n_iterations is None else data_train_loss[:n_iterations]
    data_iteration_loss = data_iteration_loss if n_iterations is None else data_iteration_loss[:n_iterations]
    n_iterations = len(data_train_loss)

    i = 0
    for i in range(2):

        if i == 0:
            """
            Plot 1: Loss Evolution after each batch
            """
            for statistic_type, statistic_label, color in [
                ("p_loss", "Policy Loss", "firebrick"),
                ("v_loss", "Value Loss", "darkorange"),
                ("loss", "Loss", "black")]:

                losses = [e[statistic_type] for e in data_train_loss]

                # smoothing
                for k, loss_list in enumerate(losses):
                    losses[k] = moving_avg(np.array(loss_list), n_batches_smooth)

                # concatenate losses to single list
                losses = [item for sublist in losses for item in sublist]

                axs[i].plot(
                    list(range(1, len(losses)+1)),
                    losses,
                    label=statistic_label,
                    color=color
                )

            batches_per_iteration = len(data_train_loss[0]["loss"])
            batches_per_iteration_after_smoothing = len(losses) // n_iterations

            # subplot title
            axs[i].set_title(f"Smoothed Batch Loss")

            # x-axis
            axs[i].set_xlim(0, len(losses))
            locs = [e*batches_per_iteration_after_smoothing for e in list(range(5, n_iterations + 1, 5))]
            labels = [e // batches_per_iteration_after_smoothing for e in locs]
            axs[i].set_xticks(locs, labels)  # Set text labels and properties.


        if i == 1:
            """
            Plot 2: Loss Evolution after each epoch
            """
            for statistic_type, statistic_label, color in [
                ("p_loss", "Policy Loss", "firebrick"),
                ("v_loss", "Value Loss", "darkorange"),
                ("loss", "Loss", "black")]:

                losses = data_iteration_loss[statistic_type]

                axs[i].plot(
                    list(range(1, len(losses)+1)),
                    losses,
                    label=statistic_label,
                    marker=".",
                    markersize=(8 if n_iterations < 50 else 6),
                    linewidth=1.5,
                    color=color
                )

            # subplot title
            axs[i].set_title(f"Evaluation Loss")

            # # x-axis
            axs[i].set_xlim(0, len(losses)+1)
            axs[i].set_xticks(list(range(5, n_iterations + 1, 5)))  # Set text labels and properties.

        # x-axis
        axs[i].set_xlabel("Training Iterations")

        # y-axis
        axs[i].set_ylabel("Loss")
        if y_max:
            axs[i].set_ylim(-0.05, y_max)

        # legend and grid
        axs[i].legend(fontsize=9)
        axs[i].grid()

    plt.subplots_adjust(left=0.08,
                        bottom=0.1,
                        right=0.99,
                        top=0.86,
                        wspace=0.21,
                        hspace=0.0)
    plt.show()
    # fig.savefig(f"./images/loss_and_results/loss_evolution_{args.game_size}x{args.game_size}.svg", format='svg', dpi=1200)


if __name__ == '__main__':

    filename_train_loss = args.directory + './train_loss.json'
    filename_iteration_loss = args.directory + './iteration_loss.json'

    with open(filename_train_loss) as f:
        data_train_loss = json.load(f)

    with open(filename_iteration_loss) as f:
        data_iteration_loss = json.load(f)

    plot(data_train_loss,
         data_iteration_loss,
         args.smooth,
         args.n_iterations,
         args.game_size,
         args.y_max)
