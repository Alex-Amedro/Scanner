"""Model versions: v1, v2... for fresh runs, v1.1, v1.2... for runs restarted from an existing version.

Each version has its own folder with the model, the VecNormalize statistics and metadata.
The periodic checkpoint overwrites the previous one; only the final save is kept.
"""

import json
import os
import re
from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import ram_guard

MODELS_ROOT = "models"


def _name_dir(name):
    return os.path.join(MODELS_ROOT, name)


def version_dir(name, version):
    return os.path.join(_name_dir(name), version)


def _version_sort_key(v):
    """Clé de tri numérique pour vN / vN.M — un tri alphabétique simple casse dès
    qu'on dépasse v9 (ex: 'v9' > 'v11' en texte, alors que v11 est postérieur)."""
    m = re.fullmatch(r"v(\d+)(?:\.(\d+))?", v)
    if not m:
        return (float("inf"), 0, v)  # motif inattendu : poussé en fin de liste, tri stable par nom
    major = int(m.group(1))
    minor = int(m.group(2)) if m.group(2) else 0
    return (major, minor)


def list_versions(name):
    d = _name_dir(name)
    if not os.path.isdir(d):
        return []
    return sorted(os.listdir(d), key=_version_sort_key)


def _major_numbers(versions):
    return sorted(int(v[1:]) for v in versions if re.fullmatch(r"v\d+", v))


def next_major_version(name):
    majors = _major_numbers(list_versions(name))
    return f"v{(majors[-1] + 1) if majors else 1}"


def next_minor_version(name, base_version):
    prefix = base_version if "." not in base_version else base_version.split(".")[0]
    existing = [v for v in list_versions(name) if v == prefix or v.startswith(prefix + ".")]
    minors = [int(v.split(".")[1]) for v in existing if "." in v]
    return f"{prefix}.{(max(minors) + 1) if minors else 1}"


def latest_version(name):
    versions = list_versions(name)
    return versions[-1] if versions else None


def checkpoint_paths(name, version):
    d = version_dir(name, version)
    return {
        "dir": d,
        "model": os.path.join(d, "model.zip"),
        "vecnormalize": os.path.join(d, "vecnormalize.pkl"),
        "metadata": os.path.join(d, "metadata.json"),
    }


def read_metadata(name, version):
    paths = checkpoint_paths(name, version)
    if not os.path.exists(paths["metadata"]):
        return {"total_timesteps": 0}
    with open(paths["metadata"]) as f:
        return json.load(f)


def save_checkpoint(name, version, model, vec_normalize_env, total_timesteps, extra_meta=None):
    paths = checkpoint_paths(name, version)
    os.makedirs(paths["dir"], exist_ok=True)
    model.save(paths["model"])
    if vec_normalize_env is not None:
        vec_normalize_env.save(paths["vecnormalize"])

    meta = {"name": name, "version": version, "total_timesteps": total_timesteps,
            "last_saved": datetime.now().isoformat(timespec="seconds")}
    if extra_meta:
        meta.update(extra_meta)
    with open(paths["metadata"], "w") as f:
        json.dump(meta, f, indent=2)


class RotatingCheckpointCallback(BaseCallback):
    """Sauvegarde périodique dans la même version (écrase le checkpoint
    précédent — pas d'accumulation de fichiers sur le disque)."""

    def __init__(self, name, version, vec_normalize_env, save_freq_steps, extra_meta=None, verbose=0):
        super().__init__(verbose)
        self.name = name
        self.version = version
        self.vec_normalize_env = vec_normalize_env
        self.save_freq_steps = save_freq_steps
        self.extra_meta = extra_meta or {}
        self._base_timesteps = 0  # timesteps déjà entraînés avant ce run (si reprise)

    def _on_training_start(self):
        self._base_timesteps = self.extra_meta.get("base_timesteps", 0)

    def _on_step(self):
        if self.n_calls % max(self.save_freq_steps // self.training_env.num_envs, 1) == 0:
            total = self.num_timesteps  # cumulatif, y compris après --resume (déjà restauré par PPO.load)
            save_checkpoint(self.name, self.version, self.model, self.vec_normalize_env, total,
                            extra_meta=dict(self.extra_meta, base_timesteps=self._base_timesteps))
            if self.verbose:
                print(f"[checkpoint] {self.name}/{self.version} sauvegardé à {total} steps")
        return True


class RamGuardCallback(BaseCallback):
    """Arrête l'entraînement PROPREMENT (return False -> learn() rend la main, train.py sauvegarde
    puis libère tout) quand la RAM libre du système passe sous min_free_gb, ou quand l'ensemble
    processus principal + workers dépasse max_tree_gb. Logue aussi la RAM dans tensorboard
    (ram/*) : une courbe qui monte sans s'arrêter d'un rollout à l'autre = fuite."""

    def __init__(self, min_free_gb, max_tree_gb=None, check_every=25, verbose=1):
        super().__init__(verbose)
        self.min_free_gb = min_free_gb
        self.max_tree_gb = max_tree_gb
        self.check_every = check_every
        self.stop_reason = None

    def _on_step(self):
        if self.n_calls % self.check_every:
            return True
        free = ram_guard.available_gb()
        if free < self.min_free_gb:
            self.stop_reason = (f"RAM libre tombée à {free:.2f} Go (seuil {self.min_free_gb:.2f} Go) "
                                f"vers {self.num_timesteps} steps")
        elif self.max_tree_gb is not None and ram_guard.tree_rss_gb() > self.max_tree_gb:
            self.stop_reason = (f"RAM de l'entraînement > {self.max_tree_gb:.2f} Go "
                                f"vers {self.num_timesteps} steps")
        if self.stop_reason:
            if self.verbose:
                print(f"\n[ram] {self.stop_reason} -> arrêt propre, sauvegarde en cours.")
            return False
        return True

    def _on_rollout_end(self):
        self.logger.record("ram/system_available_gb", ram_guard.available_gb())
        self.logger.record("ram/training_tree_gb", ram_guard.tree_rss_gb())
        self.logger.record("ram/main_process_gb", ram_guard.psutil.Process().memory_info().rss / ram_guard.GB)


class EpisodeMetricsCallback(BaseCallback):
    """Logue dans tensorboard, à chaque fin de rollout, la répartition des
    issues d'épisode (victoire / collision / retourné / timeout) et la
    couverture moyenne — pour voir en direct si les fixes (gear, détection de
    retournement) font effet, sans attendre la fin de l'entraînement."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.outcomes = []
        self.coverages = []

    def _on_step(self):
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                if info.get("retourne"):
                    outcome = "retourne"
                elif info.get("collision"):
                    outcome = "collision"
                elif info.get("truncated"):
                    outcome = "timeout"
                else:
                    outcome = "victoire"
                self.outcomes.append(outcome)
                self.coverages.append(info.get("coverage", 0.0))
        return True

    def _on_rollout_end(self):
        if not self.outcomes:
            return
        n = len(self.outcomes)
        for name in ("victoire", "collision", "retourne", "timeout"):
            rate = sum(o == name for o in self.outcomes) / n
            self.logger.record(f"episodes/{name}_rate", rate)
        self.logger.record("episodes/coverage_mean", float(np.mean(self.coverages)))
        self.logger.record("episodes/n_episodes_in_rollout", n)
        self.outcomes = []
        self.coverages = []