import json
import os
import time

import numpy as np


class Checkpoint:

    def __init__(self, checkpoint_folder: str):

        self.checkpoint_folder = checkpoint_folder
        os.makedirs(checkpoint_folder, exist_ok=True)

        self.config = checkpoint_folder + "config.json"
        self.model = checkpoint_folder + "model.pt"
        self.evaluation_results = checkpoint_folder + "results.json"
        self.train_losses = checkpoint_folder + "train_loss.json"
        self.iteration_losses = checkpoint_folder + "iteration_loss.json"
        self.training_examples = checkpoint_folder + "training_examples.json"

    def is_new_training(self):
        return not os.path.isfile(self.model)

    """
    config
    """
    def load_config(self):
        with open(self.config, 'r') as f:
            config = json.load(f)
        return config

    def save_config(self, config):
        with open(self.config, 'w') as f:
            json.dump(config, f)

    """
    evaluation_results
    """
    def load_evaluation_results(self):
        with open(self.evaluation_results, 'r') as f:
            evaluation_results = json.load(f)
        return evaluation_results

    def save_evaluation_results(self, evaluation_results):
        with open(self.evaluation_results, 'w') as f:
            json.dump(evaluation_results, f)

    """
    train_losses
    """
    def load_train_losses(self):
        with open(self.train_losses, 'r') as f:
            train_losses = json.load(f)
        return train_losses

    def save_train_losses(self, train_losses):
        with open(self.train_losses, 'w') as f:
            json.dump(train_losses, f)

    """
    iteration_losses
    """
    def load_iteration_losses(self):
        with open(self.iteration_losses, 'r') as f:
            iteration_losses = json.load(f)
        return iteration_losses

    def save_iteration_losses(self, iteration_losses):
        with open(self.iteration_losses, 'w') as f:
            json.dump(iteration_losses, f)

    """
    train_examples
    """
    def save_train_examples(self, train_examples_per_game: list):

        start_time = time.time()
        print("Saving training examples .. ", end="")

        save_dict = {}
        for i, train_examples in enumerate(train_examples_per_game):
            save_dict[i] = [{
                "lines": t[0].tolist(),
                "boxes": t[1].tolist(),
                "p": t[2].tolist(),
                "v": t[3]
            } for t in train_examples]

        with open(self.training_examples, 'w') as f:
            json.dump(save_dict, f)

        print("took {0:.2f}s".format(time.time() - start_time))

    def load_train_examples(self):

        print("Loading training examples .. ", end="")
        start_time = time.time()

        with open(self.training_examples, 'r') as f:
            save_dict = json.load(f)

        train_examples_per_game = []
        for game_id in save_dict:
            train_examples = [(
                np.array(t["lines"]),
                np.array(t["boxes"]),
                np.array(t["p"]),
                t["v"]
            ) for t in save_dict[game_id]]
            train_examples_per_game.append(train_examples)

        print("took {0:.2f}s".format(time.time() - start_time))

        return train_examples_per_game
