import copy

import numpy as np
import torch

FLOOR_GEOM_ID = 0


def rollout(env, actor, seed, state_dim, foot_geom_ids, bd_decimals, replay_buffers=None, archive=None, descriptor=None):
    """
    Runs one episode of the team policy `actor` (one head per agent) and returns (num_steps, fitness, bd).

    The behavior descriptor bd is, for each geom in foot_geom_ids, the fraction of timesteps it touches the
    floor, rounded to bd_decimals decimals. Fitness is the episode return averaged over agents.

    If replay_buffers is given, each agent's transitions are added to its buffer with d = bd and
    d_prime = descriptor (or bd when no descriptor is given). If archive is given, the policy is added to it.
    """
    observations, _ = env.reset(seed=seed)
    num_agents = env.num_agents
    agent_idx = {name: index for index, name in enumerate(env.possible_agents)}

    unwrapped_env = env.single_agent_env.unwrapped

    contact_counts = [0] * len(foot_geom_ids)
    num_steps = 0
    env_global_reward = 0.0
    experiences = [[] for _ in range(num_agents)]

    while env.agents:
        actions = {}
        for agent in env.agents:
            obs = observations[agent]

            # Agents that observe fewer than state_dim values (e.g. some HalfCheetah 6x1 agents) are zero-padded
            if obs.shape[0] < state_dim:
                obs = np.pad(obs, (0, state_dim - obs.shape[0]), 'constant', constant_values=0.0)
                observations[agent] = obs

            act = actor(torch.FloatTensor(obs).unsqueeze(0), head=agent_idx[agent])
            actions[agent] = act.cpu().data.numpy().flatten().clip(-1.0, 1.0).astype(np.float32)

        old_observations = observations.copy()

        observations, rewards, terminations, truncations, _ = env.step(actions)

        env_global_reward += sum(rewards.values()) / num_agents

        # Check which feet are touching the floor this timestep
        in_contact = [False] * len(foot_geom_ids)
        for contact in unwrapped_env.data.contact:
            for foot, foot_geom_id in enumerate(foot_geom_ids):
                if (contact.geom1 == foot_geom_id and contact.geom2 == FLOOR_GEOM_ID) or (contact.geom2 == foot_geom_id and contact.geom1 == FLOOR_GEOM_ID):
                    in_contact[foot] = True

            # Early exit if all feet are in contact
            if all(in_contact):
                break

        for foot, touching in enumerate(in_contact):
            if touching:
                contact_counts[foot] += 1

        for agent in old_observations:
            done = terminations[agent] or truncations[agent]

            next_obs = observations[agent].copy()
            if next_obs.shape[0] < state_dim:
                next_obs = np.pad(next_obs, (0, state_dim - next_obs.shape[0]), 'constant', constant_values=0.0)

            experiences[agent_idx[agent]].append((old_observations[agent], actions[agent], next_obs, rewards[agent], done))

        num_steps += 1

    bd = tuple(round(count / num_steps, bd_decimals) for count in contact_counts)

    if replay_buffers is not None:
        d_prime = bd if descriptor is None else descriptor
        for agent_id, agent_exp in enumerate(experiences):
            for state, action, next_state, reward, done in agent_exp:
                replay_buffers[agent_id].add(state, action, next_state, reward, done, d_prime=d_prime, d=bd)

    if archive is not None:
        archive.add(policy=copy.deepcopy(actor), bd=bd, fitness=env_global_reward)

    return num_steps, env_global_reward, bd
