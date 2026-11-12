"""
Train MASQD on a multiagent MuJoCo task, e.g.

    python -m masqd.train --config configs/walker2d_2x3.yaml --seed 2025
"""
import argparse
import os
import random

import numpy as np
import pypickle
import torch
import yaml

from .archive import BehaviorArchive
from .config import load_config
from .dc_td3 import DCTD3
from .envs import make_env
from .masqd import run_masqd
from .networks import Actor
from .replay_buffer import ReplayBuffer


def save_results(results_dir, run_name, archive, qd_scores, max_fitnesses, archive_sizes):
    """Pickles the final archive and the (env steps, value) histories recorded every iteration."""
    os.makedirs(results_dir, exist_ok=True)
    outputs = {
        "archive": archive,
        "qdscore": qd_scores,
        "maxfit": max_fitnesses,
        "archivesize": archive_sizes,
    }
    for name, obj in outputs.items():
        path = os.path.join(results_dir, f"{run_name}_{name}.pkl")
        # allow_external: pypickle otherwise refuses paths outside the home and temp directories
        if not pypickle.save(path, obj, overwrite=True, allow_external=True):
            raise RuntimeError(f"pypickle could not save {path}")


def main():
    parser = argparse.ArgumentParser(description="Run MASQD on a multiagent MuJoCo task.")
    parser.add_argument("--config", required=True, help="YAML config, e.g. configs/walker2d_2x3.yaml")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-iterations", type=int, default=None, help="Stop after this many iterations (default: run until the env step budget is used up)")
    parser.add_argument("--log-dir", default="logs")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    env_config = config["env"]

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Start each run with an empty log file, and save the config it ran with next to it
    run_name = f"{env_config['name']}_seed{args.seed}"
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, run_name + ".log")
    open(log_path, "w").close()
    with open(os.path.join(args.log_dir, run_name + "_config.yaml"), "w") as f:
        yaml.safe_dump({**config, "run": vars(args)}, f, sort_keys=False)

    env = make_env(env_config)
    state_dim = env_config["state_dim"]
    action_dim = env_config["action_dim"]
    num_agents = env_config["num_agents"]
    num_bds = len(env_config["foot_geom_ids"])
    networks = config["networks"]

    # NOTE: the actor and the TD3 networks draw their initial weights from the seeded random sequence, so creating
    # them in a different order changes the results for a given seed.
    actor = Actor(state_dim, action_dim, networks["actor_hidden_size"], num_heads=num_agents)
    archive = BehaviorArchive()
    replay_buffers = [ReplayBuffer(state_dim, action_dim, num_bds, max_size=config["replay_buffer"]["max_size"]) for _ in range(num_agents)]
    td3 = DCTD3(
        state_dim,
        action_dim,
        num_agents,
        num_bds,
        args.device,
        networks["actor_hidden_size"],
        networks["critic_hidden_size"],
        **config["dc_td3"],
    )

    qd_scores, max_fitnesses, archive_sizes = run_masqd(
        env,
        env_config,
        actor,
        td3,
        archive,
        replay_buffers,
        seed=args.seed,
        log_path=log_path,
        device=args.device,
        max_iterations=args.max_iterations,
        **config["masqd"],
    )

    save_results(args.results_dir, run_name, archive, qd_scores, max_fitnesses, archive_sizes)


if __name__ == "__main__":
    main()
