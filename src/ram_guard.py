"""RAM guard for training: estimate the memory before launch (to choose --n-envs), measure it during the run, clean up the worker processes at the end.

It does not import torch or SB3 on purpose: train.py imports this module at top level, and Windows imports it again in every SubprocVecEnv worker.
"""

import gc

import numpy as np

try:
    import psutil
except ImportError:  # le garde-fou se désactive, l'entraînement continue
    psutil = None

GB = 2 ** 30

# Constantes calibrées sur la baisse réelle de RAM DISPONIBLE du système pendant un entraînement
# (CPU, 8 envs : ~3,3 Go ; CUDA, 10 envs : ~6,5 Go), PAS sur la RAM "résidente" des processus : elle
# compte plusieurs fois les bibliothèques partagées entre workers (~1,5x trop haut, d'où une
# première calibration trop pessimiste, qui refusait de lancer 1 seul env avec 6 Go libres).
# Coût d'un worker SubprocVecEnv (mujoco + gymnasium + numpy, sans torch) : ~70-100 MB.
# Marge large pour les pics de reset (génération du bâtiment, MjModel).
WORKER_GB = 0.22
# Processus principal une fois torch/SB3 importés, hors rollout buffer (+ contexte CUDA côté hôte).
MAIN_BASE_GB_CPU = 1.2
MAIN_BASE_GB_CUDA = 3.7
# Le rollout buffer de SB3 stocke l'observation complète de chaque step, plus des copies transitoires.
BUFFER_OVERHEAD = 1.5


def available() -> bool:
    return psutil is not None


def available_gb() -> float:
    return psutil.virtual_memory().available / GB


def total_gb() -> float:
    return psutil.virtual_memory().total / GB


def tree_rss_gb() -> float:
    """RAM résidente du processus courant + de tous ses workers."""
    me = psutil.Process()
    total = me.memory_info().rss
    for child in me.children(recursive=True):
        try:
            total += child.memory_info().rss
        except psutil.Error:
            pass
    return total / GB


def obs_bytes(observation_space) -> int:
    return int(sum(int(np.prod(s.shape)) * 4 for s in observation_space.spaces.values()))


def estimate_run_gb(n_envs, n_steps, observation_space, cuda) -> float:
    buffer_gb = n_envs * n_steps * obs_bytes(observation_space) * BUFFER_OVERHEAD / GB
    base = MAIN_BASE_GB_CUDA if cuda else MAIN_BASE_GB_CPU
    return base + n_envs * WORKER_GB + buffer_gb


def choose_n_envs(requested, n_steps, observation_space, cuda, reserve_gb):
    """Plus grand n_envs <= requested dont l'estimation tient dans (RAM disponible - réserve).
    Renvoie (n_envs, estimation_GB, budget_GB). Jamais moins de 1."""
    budget = available_gb() - reserve_gb
    n = requested
    while n > 1 and estimate_run_gb(n, n_steps, observation_space, cuda) > budget:
        n -= 1
    return n, estimate_run_gb(n, n_steps, observation_space, cuda), budget


def kill_leftover_children(timeout=3.0):
    """Termine les processus enfants encore vivants (workers non fermés proprement)."""
    if psutil is None:
        return 0
    children = psutil.Process().children(recursive=True)
    for c in children:
        try:
            c.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(children, timeout=timeout)
    for c in alive:
        try:
            c.kill()
        except psutil.Error:
            pass
    return len(children)


def release(env=None):
    """Ferme l'env vectorisé (workers inclus), vide le cache CUDA et tue ce qui traîne (l'appelant
    doit avoir lâché ses propres références au modèle pour que gc.collect() serve à quelque chose). À appeler dans un `finally` : doit marcher même après une erreur de mémoire."""
    if env is not None:
        try:
            env.close()
        except Exception as e:  # noqa: BLE001 - on veut tout nettoyer quoi qu'il arrive
            print(f"[ram] fermeture de l'env : {type(e).__name__}: {e}")
    n_left = kill_leftover_children()
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass
    if n_left:
        print(f"[ram] {n_left} processus enfant(s) encore actif(s) après close() : terminés.")
