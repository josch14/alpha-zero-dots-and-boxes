

# AlphaZero: Dots and Boxes
**AlphaZero** implementation for the Pen and Paper game **Dots and Boxes** (Project Deep Reinforcement Learning @ Universität Ulm). 

Have a look at the full report [here](0_report.pdf), providing detailed information about the AlphaZero algorithm, how it is adapted for Dots and Boxes, and visualizations of training progress including loss and results evolution against other game-playing AI approaches.

## Abstract
The introduction of AlphaZero in 2017 was a milestone in the field of game-playing artificial intelligence. Until then, development of the strongest programs was based on game-specific search techniques, adaptations, and handcrafted evaluations created by human experts. In contrast, AlphaZero learns and masters board games by reinforcement learning from self-play without human guidance beyond game rules, reaching superhuman performance for complex board games such as chess, shogi and Go. In this work, we apply the AlphaZero algorithm to the game of Dots and Boxes. In this context, we analyze the training process of AlphaZero and evaluate its performance against other artificial intelligence based game-playing programs for small board sizes. Further, we discuss the challenges and requirements involved in successfully applying AlphaZero to other board games. While showing its forward-looking capabilities, AlphaZero consistently beats its opponents in our experiments.





## Features

* [x] Train (and play against) AlphaZero for the game of Dots and Boxes
* [x] Providing detailed information during training
```
#################### Iteration 21/1000 ####################
------------ Self-Play using MCTS ------------
100%|██████████| 500/500 [1:33:38<00:00, 11.24s/it]
500 games of Self-Play resulted in 19,423 new training examples (without augmentations).
Loading training examples .. took 75.63s
Saving training examples .. took 434.59s

---------- Neural Network Training -----------
Encoding train examples for given model ..
Batches are sampled from 1,549,200 training examples (incl. augmentations) from the 5,000/5,000 most recent games.
Preparing batches ..
100%|██████████| 5600/5600 [00:15<00:00, 370.57it/s]
Updating model ..
100%|██████████| 5600/5600 [07:29<00:00, 12.47it/s]
Evaluating model ..
100%|██████████| 5600/5600 [02:11<00:00, 42.53it/s]
Policy Loss: 3.03951 (avg.)
Value Loss: 0.33518 (avg.)
Loss: 3.37469 (avg.)

-------------- Model Comparison --------------
Comparing UpdatedModel:Draw:RandomPlayer ...
100%|██████████| 50/50 [04:24<00:00,  5.29s/it]
100%|██████████| 50/50 [04:41<00:00,  5.63s/it]
Result: 50:0:0 (starting)
Result: 50:0:0 (second)
Result: 100:0:0 (total)

...

Comparing UpdatedModel:Draw:AlphaBetaPlayer(Depth=3) ...
100%|██████████| 50/50 [05:50<00:00,  7.01s/it]
100%|██████████| 50/50 [06:03<00:00,  7.26s/it]
Result: 46:3:1 (starting)
Result: 28:9:13 (second)
Result: 74:12:14 (total)

Total time in training loop: 75695.07s
###########################################################
```

* [x] Play Dots and Boxes versus AlphaZero (or other opponents) in terminal 

TODO img

* [x] Visualization of loss evolution

TODO img

* [x] Visualization of results evolution against other game-playing AI approaches

TODO img

* [x] Multi-thread support for self-play
* [x] GPU support for training and self-play
* [x] Extensively documented code
* [x] Model weights, statistics and training data automatically saved at checkpoints
* [x] Single and two player mode


## Improvements (To-Do)
* [ ] Print AlphaZero search probabilities for moves when playing in terminal
* [ ] Modify play.py to view games of AlphaZero(s) (or different AIs in general) playing against each other
* [ ] Multi GPU support for the training and the selfplay
* [ ] Batch MCTS




## Installation
```bash
conda create -n azero_dab python=3.8
conda activate azero_dab

# PyTorch: select between cpu only and cuda support
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

conda install -c anaconda pyyaml  # load config files
conda install -c conda-forge tqdm  # training progress
conda install -c conda-forge matplotlib  # visualizations after training

# enable colored print when playing in console
conda install -c conda-forge termcolor=1.1.0
conda install -c anaconda colorama
```







## Training

Training (requires a config file) can be started from scratch or continued from checkpoint. We use gpu support only during neural network update (`--training_device={cpu, cuda}`). When using a larger neural network, gpu support might be beneficial during self-play (`--inference_device={cpu, cuda}`). You may increase the number of workers to utilize multiprocessing.

### Training from scratch
```bash
python train.py --config 2x2_dual_res --n_workers 4 --inference_device cpu --training_device cuda
```

### Logs
Running training will create a timestamp-named folder under `logs` including the config, loss and results statistics, trained neural network, and a file with recent training data.

### Continue from Checkpoint
To continue training from checkpoint, specify the name of the corresponding logs folder (rename the timestamp before!). We provide the logs from our AlphaZero training on the 2x2, 3x3 and 4x4 board. However, this does not include recent training data (file's too large). Therefore, training can not be continued for these models, unless you want to use the models as pre-trained models for new training. 
```bash
python train.py --checkpoint alpha_zero_2x2 --n_workers 4 --inference_device cpu --training_device cuda
```

### Config
We provide the config files that we used (`2x2_dual_res.yaml`, `3x3_dual_res.yaml` and `2x2_dual_res.yaml`). Modify those with respect to your computational power and target board size. If you want to employ a neural network with simpler architecture, have a look at `2x2_feed_forward.yaml`. 






## Play Dots and Boxes
You can play Dots and Boxes in the terminal against different opponents: a second person, against AlphaZero (requires specifying a checkpoint), a `RandomPlayer`, and a `AlphaBetaPlayer` (requires specifying a search depth).

```bash
python play.py --size 3 --opponent person
python play.py --size 3 --opponent alpha_zero --checkpoint alpha_zero_3x3
python play.py --size 3 --opponent random
python play.py --size 3 --opponent alpha_beta --depth 3
```


## Plots
You may visualize the training progress by plotting the loss evolution and results evolution.
```bash
python plot_loss.py -d '.\logs\alpha_zero_3x3' -s 3
python plot_results.py -d '.\logs\alpha_zero_3x3' -s 3
```