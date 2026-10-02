"""Extracteur de features pour PPO : un petit CNN sur le crop local (grille
d'occupation, mémoire spatiale) + un petit MLP sur (frontières + rayons de
relief + cinématique + couverture), concaténés en un vecteur latent unique.
"""

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ExplorerFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, cnn_out_dim=128, mlp_out_dim=64):
        super().__init__(observation_space, features_dim=cnn_out_dim + mlp_out_dim)

        crop_shape = observation_space["local_crop"].shape  # (3, H, W)
        n_channels = crop_shape[0]

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 16, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=1), nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, *crop_shape)
            cnn_flat_dim = self.cnn(dummy).shape[1]
        self.cnn_head = nn.Sequential(nn.Linear(cnn_flat_dim, cnn_out_dim), nn.ReLU())

        self.scalar_keys = ["frontier_vector", "relief_rays", "kinematics", "coverage", "proximity_rays", "last_action"]
        scalar_dim = sum(observation_space[k].shape[0] for k in self.scalar_keys)
        self.mlp = nn.Sequential(
            nn.Linear(scalar_dim, 64), nn.ReLU(),
            nn.Linear(64, mlp_out_dim), nn.ReLU(),
        )

    def forward(self, observations):
        crop_features = self.cnn_head(self.cnn(observations["local_crop"]))
        scalars = torch.cat([observations[k] for k in self.scalar_keys], dim=1)
        scalar_features = self.mlp(scalars)
        return torch.cat([crop_features, scalar_features], dim=1)