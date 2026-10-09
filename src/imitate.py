"""Imitate the scripted explorer (expert.py) with the PPO policy network (MLP + CNN).

  1. The expert plays on houses of several levels. Noise is added to the executed action; the clean expert action is recorded.
  2. The network learns to reproduce the expert's mean action (MSE). Input normalisation comes from the same data.
  3. The model is saved like the others (evaluate it with evaluate.py, --max-steps 3000 for large houses).

Usage:
    python imitate.py --name bc2_s42 --levels 1 4 --samples 80000 --epochs 25
"""
import argparse
import json
import sys

import numpy as np
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import model_manager as mm
from caps_ppo import PPOCaps
from expert import EXPERT_ENV_KW, expert_action
from explorer_env import ExplorerEnv
from feature_extractor import ExplorerFeaturesExtractor

sys.stdout.reconfigure(errors="replace")
NORM_KEYS = ["frontier_vector", "proximity_rays", "velocity"]

ap = argparse.ArgumentParser()
ap.add_argument("--name", default="bc_s42")
ap.add_argument("--base", default=None, help="(Facultatif) modèle dont on reprend les réglages d'environnement ; sinon EXPERT_ENV_KW de expert.py.")
ap.add_argument("--levels", type=int, nargs=2, default=[1, 4])
ap.add_argument("--samples", type=int, default=60000)
ap.add_argument("--every", type=int, default=2, help="Enregistre un pas sur N.")
ap.add_argument("--noises", type=float, nargs="+", default=[0.0, 0.15, 0.3], help="Bruit (écart-type) sur l'action exécutée, tiré par épisode.")
ap.add_argument("--speed-frac", type=float, default=0.6)
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max-steps", type=int, default=3000)
ap.add_argument("--log-std", type=float, default=-1.5, help="log(écart-type) initial de la politique pour un affinage PPO.")
ap.add_argument("--dry", action="store_true", help="Test : ne sauvegarde rien.")
args = ap.parse_args()

rng = np.random.default_rng(args.seed)
th.manual_seed(args.seed)
base = mm.read_metadata(args.base, mm.latest_version(args.base))["env_kwargs"] if args.base else EXPERT_ENV_KW
kw = dict(base)
kw.update(building="house", house_level=tuple(args.levels), frontier_mode="bfs", bfs_clearance_m=0.45, max_steps=args.max_steps,
          obs_keys=["proximity_rays", "frontier_vector", "velocity", "local_crop"])
kw["n_rooms"] = tuple(kw.get("n_rooms", (3, 3)))
print("réglages d'environnement :", {k: kw[k] for k in ("building", "house_level", "frontier_mode", "bfs_clearance_m", "max_steps", "obs_keys", "no_yaw")})

# ---------------------------------------------------------------- 1. collecte
env = ExplorerEnv(seed=args.seed, **kw)
raw = {k: [] for k in ("proximity_rays", "frontier_vector", "velocity", "local_crop")}
acts = []
ep, step_total = 0, 0
wins = 0
while len(acts) < args.samples:
    sigma = float(rng.choice(args.noises))
    obs, _ = env.reset(seed=1000 + ep)          # seeds d'entraînement, distinctes des seeds d'évaluation (9000+)
    for t in range(args.max_steps):
        a_exp = expert_action(obs, args.speed_frac)
        if t % args.every == 0:
            for k in raw:
                raw[k].append(obs[k].astype(np.uint8) if k == "local_crop" else obs[k].astype(np.float32))
            acts.append(a_exp)
        a_exec = np.clip(a_exp + sigma * rng.standard_normal(2), -1, 1).astype(np.float32)
        obs, r, term, trunc, info = env.step(a_exec)
        step_total += 1
        if term or trunc:
            break
    wins += int(env.grid.coverage_ratio() >= 0.9 and term)
    ep += 1
    if ep % 20 == 0:
        print(f"  {ep} épisodes, {len(acts)}/{args.samples} échantillons, {step_total} pas simulés, {wins} épisodes finis à 90 %")
env.close()
data = {k: np.stack(v) for k, v in raw.items()}
acts = np.stack(acts).astype(np.float32)
print(f"collecte finie : {len(acts)} échantillons sur {ep} épisodes ; action experte moyenne |a| = {np.abs(acts).mean():.2f}")

# ---------------------------------------------------------------- 2. réseau + normalisation
vec = DummyVecEnv([lambda: ExplorerEnv(seed=args.seed, **kw)])
vn = VecNormalize(vec, norm_obs=True, norm_obs_keys=NORM_KEYS, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
for k in NORM_KEYS:
    vn.obs_rms[k].update(data[k].astype(np.float64))
policy_kwargs = dict(features_extractor_class=ExplorerFeaturesExtractor, features_extractor_kwargs=dict(cnn_out_dim=128, mlp_out_dim=64),
                     net_arch=dict(pi=[128], vf=[128]))
model = PPOCaps("MultiInputPolicy", vn, policy_kwargs=policy_kwargs, caps_lambda=1.0, n_steps=1024, batch_size=256, seed=args.seed,
                device="cpu", gamma=0.99, verbose=0)
model.policy.log_std.data.fill_(args.log_std)

def batch_obs(idx):
    d = {k: data[k][idx].astype(np.float32) for k in data}
    d = vn.normalize_obs(d)
    return {k: th.as_tensor(v, dtype=th.float32) for k, v in d.items()}

# ---------------------------------------------------------------- 3. clonage
n = len(acts); perm = rng.permutation(n); n_val = n // 10
val_idx, tr_idx = perm[:n_val], perm[n_val:]
opt = th.optim.Adam(model.policy.parameters(), lr=args.lr)
model.policy.set_training_mode(True)
for epoch in range(args.epochs):
    rng.shuffle(tr_idx)
    tr_loss = []
    for i in range(0, len(tr_idx), 256):
        idx = tr_idx[i:i + 256]
        mu = model.policy.get_distribution(batch_obs(idx)).distribution.mean
        loss = ((mu - th.as_tensor(acts[idx])) ** 2).mean()
        opt.zero_grad(); loss.backward(); th.nn.utils.clip_grad_norm_(model.policy.parameters(), 0.5); opt.step()
        tr_loss.append(loss.item())
    with th.no_grad():
        vl = []
        for i in range(0, len(val_idx), 512):
            idx = val_idx[i:i + 512]
            mu = model.policy.get_distribution(batch_obs(idx)).distribution.mean
            vl.append(((mu - th.as_tensor(acts[idx])) ** 2).mean().item())
    print(f"époque {epoch + 1:2d}/{args.epochs}  MSE entraînement {np.mean(tr_loss):.4f}  validation {np.mean(vl):.4f}  (variance des actions expertes {acts.var():.4f})")

if args.dry:
    print("--dry : rien sauvegardé"); sys.exit(0)
version = mm.next_major_version(args.name)
mm.save_checkpoint(args.name, version, model, vn, 0, extra_meta={"env_kwargs": {k: (list(v) if isinstance(v, tuple) else v) for k, v in kw.items()},
                                                                  "imitation": {"expert_speed_frac": args.speed_frac, "samples": int(n), "noises": args.noises}})
print(f"modèle sauvegardé : {args.name}/{version}")
