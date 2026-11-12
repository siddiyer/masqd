# Multiagent Sample-Efficient Quality-Diversity (MASQD)

This repo contains the code for "Multiagent Quality-Diversity for Effective Adaptation," published at ECAI 2025.

MASQD learns an archive of diverse, high-performing multiagent teams. It builds on the Cross-Entropy Method Reinforcement Learning (CEM-RL) algorithm and extends it to the multiagent setting by representing teams as parameter-shared neural networks, directing exploration from previously discovered behaviors, and guiding refinement through a descriptor-conditioned critic.

## Installation

Requires Python 3.10.16. A CUDA GPU is used if available.

```bash
conda create -n masqd python=3.10.16
conda activate masqd
pip install -r requirements.txt
```

## Training

```bash
python -m masqd.train --config configs/walker2d_2x3.yaml --seed 2025
python -m masqd.train --config configs/halfcheetah_2x3.yaml --seed 2025
python -m masqd.train --config configs/halfcheetah_6x1.yaml --seed 2025
python -m masqd.train --config configs/hopper_3x1.yaml --seed 2025
```

Seeds [2025, 2029] were used for training in the paper.

Optional arguments: `--max-iterations N` (stop early instead of at the config's env-step budget), `--device`, `--log-dir` (default `logs`), `--results-dir` (default `results`).

Each run writes:

- `logs/<env>_seed<seed>.log`: one line per iteration with env steps, QD score, max fitness and archive size.
- `logs/<env>_seed<seed>_config.yaml`: the config and arguments the run used.
- `results/<env>_seed<seed>_{archive,qdscore,maxfit,archivesize}.pkl`, at the end of the run: the final archive and the `(env steps, value)` history of each metric.

To load the archive (from the repo root, so `masqd` can be imported):

```python
import pypickle

archive = pypickle.load("results/walker2d_seed2025_archive.pkl")
for bd, (fitness, policy) in archive.archive_dict.items():
    ...  # policy(obs, head=i) gives agent i's action
```

## Configuration

Each environment has one YAML file in `configs/` holding every setting: the environment and behavior descriptor (`env`), the MASQD loop (`masqd`), network sizes (`networks`), the descriptor-conditioned TD3 (`dc_td3`) and the replay buffer size. The behavior descriptor is the fraction of timesteps each foot geom in `env.foot_geom_ids` touches the floor.

## Code

- `masqd/train.py`: entry point
- `masqd/masqd.py`: main loop (warm-up, re-anchoring on the archive, CEM sampling and update, refinement)
- `masqd/dc_td3.py`: descriptor-conditioned TD3 (critic training and policy refinement)
- `masqd/rollout.py`: episode rollout and behavior descriptor
- `masqd/networks.py`, `archive.py`, `replay_buffer.py`, `envs.py`, `config.py`: networks, archive, per-agent replay buffers, environment creation, config loading

## BibTeX

```
@inproceedings{iyer2025multiagent,
  title={Multiagent Quality-Diversity for Effective Adaptation.},
  author={Iyer, Siddarth and Aydeniz, Ayhan Alp and Dixit, Gaurav and Tumer, Kagan},
  booktitle={ECAI},
  pages={3323--3330},
  year={2025}
}
```
