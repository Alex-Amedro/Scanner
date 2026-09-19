"""Diagnostic (gratuit, pas de réentraînement) : lit vecnormalize.pkl d'une version pour mesurer
l'ampleur de la normalisation du reward (VecNormalize.ret_rms) — sert à vérifier l'hypothèse que
les signaux rares et forts (collision -50, bonus de victoire r_exp) sont écrasés par l'écart-type
courant des retours cumulés avant d'atteindre le réseau, alors que le flux dense de petits rewards
(n_new_cells à chaque step) ne l'est presque pas.

Usage :
    python check_reward_norm.py --name explorer --version v17
    python check_reward_norm.py --name explorer --version v12   # pour comparer
"""

import argparse

import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import model_manager as mm
from explorer_env import ExplorerEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", type=str, default="explorer")
    parser.add_argument("--version", type=str, default=None)
    args = parser.parse_args()

    version = args.version or mm.latest_version(args.name)
    paths = mm.checkpoint_paths(args.name, version)
    meta = mm.read_metadata(args.name, version)
    env_kwargs = meta.get("env_kwargs", {})

    vec_env = DummyVecEnv([lambda: ExplorerEnv(n_rooms=(2, 4), seed=0, **env_kwargs)])
    vec_env = VecNormalize.load(paths["vecnormalize"], vec_env)

    ret_var = float(vec_env.ret_rms.var)
    ret_std = ret_var ** 0.5
    clip = vec_env.clip_reward

    print(f"=== {args.name}/{version} ({meta.get('total_timesteps', '?')} steps) ===")
    print(f"ret_rms.var  = {ret_var:.4f}")
    print(f"ret_rms.std  = {ret_std:.4f}   <- diviseur effectif du reward brut, avant clip")
    print(f"clip_reward  = {clip}\n")

    r_exp = env_kwargs.get("r_exp", 100.0)
    cas = [
        ("Collision (-50, fixe dans le code)", -50.0),
        (f"Victoire (r_exp={r_exp})", r_exp),
        ("Step 'normal' (~10, n_new_cells typique début d'exploration)", 10.0),
        ("Step 'faible' (~1, n_new_cells en fin d'exploration)", 1.0),
    ]

    for label, raw in cas:
        normalized = raw / ret_std
        clipped = float(np.clip(normalized, -clip, clip))
        note = ""
        if abs(normalized) > clip:
            pct_lost = 100 * (1 - abs(clipped) / abs(normalized))
            note = f"  <-- ÉCRÊTÉ, {pct_lost:.0f}% de la valeur perdue au clip"
        print(f"{label}")
        print(f"  brut       : {raw:+.2f}")
        print(f"  normalisé  : {normalized:+.4f}")
        print(f"  après clip : {clipped:+.4f}{note}\n")


if __name__ == "__main__":
    main()