# Patch-Clam-Localisation
The Official Implementation of In Vivo Patch Clamp Multi-Pipette Localisation

Developers: Lan Wei

# This code is the implementation of:
[Heatmap-Augmented Coarse-to-Fine Learning for Automatic In Vivo Patch Clamp Multi-Pipette Real-time Localisation]

Authors: Lan Wei, Gema Vera Gonzalez, Phatsimo Kgwarae, Alexander Timms, Denis Zahorovsky, Simon Schultz, and Dandan Zhang

## Usage

First, install PyTorch == 1.9.1+cu111\\
prchvision == 0.10.1+cu111 \\
tensorboardx == 2.4\\
tensorboard_logger == 0.1.0:
```
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install tensorboardx == 2.4
pip install tensorboard_logger == 0.1.0
```

# Model training
```
python Lan_model_train.py --gan_enhance 1 --eu_dist 3 --heatmap_sigma 10 --change_epoch_1 50 --change_epoch_2 100 --n_epochs 200
```

