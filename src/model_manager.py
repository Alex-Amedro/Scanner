"""Gestion des versions de modèles : v1, v2, v3... pour des entraînements
frais, v1.1, v1.2... pour repartir d'une version existante. Chaque version a
son propre dossier avec le modèle, la normalisation (VecNormalize) et des
métadonnées — le checkpoint périodique écrase le précédent (pas d'accumulation
de fichiers), seule la sauvegarde finale d'une version reste définitivement.
"""

import json
import os
import re
from datetime import datetime

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

MODELS_ROOT = "models"


def _name_dir(name):
    return os.path.join(MODELS_ROOT, name)


def version_dir(name, version):
    return os.path.join(_name_dir(name), version)


def list_versions(name):
    d = _name_dir(name)
    if not os.path.isdir(d):
        return []
    return sorted(os.listdir(d))


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
            total = self._base_timesteps + self.num_timesteps
            save_checkpoint(self.name, self.version, self.model, self.vec_normalize_env, total)
            if self.verbose:
                print(f"[checkpoint] {self.name}/{self.version} sauvegardé à {total} steps")
        return True


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