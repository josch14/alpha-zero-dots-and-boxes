# dots-and-boxes-alpha-zero
Project Deep Reinforcement Learning, Universität Ulm, WiSe 22/23. <br />
**AlphaZero** implementation for the Pen and Paper game **Dots-and-Boxes**. 

## Installation
```
conda create -n azero_dab python=3.8
conda activate azero_dab

# PyTorch
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cpuonly -c pytorch  # cpu only
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch  # gpu support

conda install -c anaconda numpy=1.23.5
conda install -c conda-forge tqdm
conda install -c anaconda pyyaml
conda install -c conda-forge matplotlib

# enable colored print for playing in console
conda install -c conda-forge termcolor=1.1.0
conda install -c anaconda colorama
```
