"""PPO + CAPS (Conditioning for Action Policy Smoothness, Mysore et al., ICRA 2021, arXiv 2012.06644), terme TEMPOREL seulement.

L_T = || mu(s_t) - mu(s_{t+1}) ||^2, où mu est l'ACTION MOYENNE (déterministe) de la politique sur deux pas CONSÉCUTIFS d'un même épisode.
Ce terme est ajouté à la perte de PPO (pas au reward) :
  - le bruit d'exploration ne peut pas le déclencher (on pénalise la moyenne, jamais l'action échantillonnée) -> impossible de le
    « gagner » en mourant tôt (c'était le défaut de la pénalité sur l'action échantillonnée) ;
  - il ne passe pas par le retour cumulé, donc la normalisation du reward ne peut pas l'écraser.

Adaptation à Stable-Baselines3 2.9.0 : la méthode train() est celle de PPO, copiée avec un seul ajout. Les paires (t, t+1) sont construites AVANT
la première lecture du tampon (get() le réorganise ensuite par environnement) et on exclut tout couple qui traverse une fin d'épisode.
"""
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F


class PPOCaps(PPO):
    def __init__(self, *args, caps_lambda=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.caps_lambda = caps_lambda

    def _build_pairs(self):
        """Paires (obs_t, obs_t+1) du rollout qui vient d'être collecté, sans traverser un reset. Renvoie deux dicts de tenseurs."""
        rb = self.rollout_buffer
        valid = rb.episode_starts[1:] == 0                      # (n_steps-1, n_envs) : t+1 n'est pas le début d'un nouvel épisode
        ts, es = np.nonzero(valid)
        obs_t, obs_t1 = {}, {}
        for key, arr in rb.observations.items():                # arr : (n_steps, n_envs, *shape)
            obs_t[key] = th.as_tensor(arr[ts, es], device=self.device).float()
            obs_t1[key] = th.as_tensor(arr[ts + 1, es], device=self.device).float()
        return obs_t, obs_t1, len(ts)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        use_caps = self.caps_lambda > 0.0
        if use_caps:
            pairs_t, pairs_t1, n_pairs = self._build_pairs()
            self._last_n_pairs = n_pairs

        entropy_losses, pg_losses, value_losses, clip_fractions, caps_losses = [], [], [], [], []
        continue_training = True
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fractions.append(th.mean((th.abs(ratio - 1) > clip_range).float()).item())

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(values - rollout_data.old_values, -clip_range_vf, clip_range_vf)
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                entropy_loss = -th.mean(-log_prob) if entropy is None else -th.mean(entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                if use_caps:
                    idx = th.randint(0, n_pairs, (min(self.batch_size, n_pairs),), device=self.device)
                    b_t = {k: v[idx] for k, v in pairs_t.items()}
                    b_t1 = {k: v[idx] for k, v in pairs_t1.items()}
                    mu_t = self.policy.get_distribution(b_t).mode()
                    mu_t1 = self.policy.get_distribution(b_t1).mode()
                    caps_loss = ((mu_t - mu_t1) ** 2).sum(dim=1).mean()
                    caps_losses.append(caps_loss.item())
                    loss = loss + self.caps_lambda * caps_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        if caps_losses:
            self.logger.record("train/caps_loss", np.mean(caps_losses))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
