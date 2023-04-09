

# AlphaZero: Dots and Boxes
**AlphaZero** implementation for the Pen and Paper game **Dots and Boxes** (Project Deep Reinforcement Learning, Universität Ulm, WiSe 22/23.). 

## Abstract
The introduction of AlphaZero in 2017 was a milestone in the field of game-playing artificial intelligence. Until then, development of the strongest programs was based on game-specific search techniques, adaptations, and handcrafted evaluations created by human experts. In contrast, AlphaZero learns and masters board games by reinforcement learning from self-play without human guidance beyond game rules, reaching superhuman performance for complex board games such as chess, shogi and Go. In this work, we apply the AlphaZero algorithm to the game of Dots and Boxes. In this context, we analyze the training process of AlphaZero and evaluate its performance against other artificial intelligence based game-playing programs for small board sizes. Further, we discuss the challenges and requirements involved in successfully applying AlphaZero to other board games. While showing its forward-looking capabilities, AlphaZero consistently beats its opponents in our experiments.

TODO link to report

## Features
TODO


## TODO
- view games between different opponents
- 

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