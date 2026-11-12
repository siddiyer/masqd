import copy
import itertools

import numpy as np
import torch

from .networks import get_flat_params, set_flat_params
from .rollout import rollout


def run_masqd(
    env,
    env_config,
    actor,
    td3,
    archive,
    replay_buffers,
    seed,
    log_path,
    device,
    max_iterations=None,
    *,
    max_env_steps,
    population_size,
    elite_frac,
    warmup_iterations,
    resample_every,
    sigma_init,
    damp_init,
    damp_limit,
    damp_tau,
    target_bd_noise,
    critic_update_interval,
    num_target_labeled,
):
    """
    Runs MASQD until max_env_steps environment steps (or max_iterations iterations) have been used.
    The keyword-only arguments are the masqd section of a config.

    Each iteration samples a population of team policies from a Gaussian (CEM). The first half is evaluated
    directly. After the warm-up, the second half is first refined towards a target behavior descriptor with the
    descriptor-conditioned critic, then evaluated. Every evaluated policy goes into the archive and its
    transitions into the replay buffers. Every resample_every iterations the Gaussian is re-centered on a
    policy sampled from the archive, and that policy's (noisy) descriptor becomes the new target.

    `actor` is used as a scratch network to evaluate sampled parameter vectors. Appends one line per iteration
    to log_path and returns the (env steps, value) histories of QD score, max fitness and archive size.
    """

    def evaluate(policy, record=True, descriptor=None):
        """Runs one episode. With record=True the policy goes into the archive and its transitions into the replay buffers."""
        steps, fitness, _ = rollout(
            env,
            policy,
            seed,
            env_config["state_dim"],
            env_config["foot_geom_ids"],
            env_config["bd_decimals"],
            replay_buffers=replay_buffers if record else None,
            archive=archive if record else None,
            descriptor=descriptor,
        )
        return fitness, steps

    # parents are the elites that are left, in this case the top half are used in the update rule
    parents = int(population_size * elite_frac)

    # weights are assigned to each parent to determine the impact in the gradient update (best parent has the most impact on the mean's change)
    weights = np.array([np.log((parents + 1) / ind) for ind in range(1, parents + 1)])
    weights /= weights.sum()

    # The first half of each population is evaluated directly, the second half is refined with RL first
    num_cem = population_size // 2

    # Initialize distribution parameters (mean and diagonal covariance)
    initial_params = get_flat_params(actor).numpy().flatten()  # flatten to 1D
    num_params = initial_params.shape[0]
    mu = initial_params.copy()
    cov = np.ones(num_params) * sigma_init  # initial variance
    damp = damp_init

    total_steps = 0

    target_steps_for_critic = None

    qd_scores = []
    max_fitnesses = []
    archive_sizes = []

    for iteration in itertools.count():
        if total_steps >= max_env_steps or (max_iterations is not None and iteration >= max_iterations):
            break

        solutions = []
        rewards = []

        if iteration >= warmup_iterations and (iteration - warmup_iterations) % resample_every == 0:
            # swap out mean + reinitialize covariance for a new policy's weights sampled from the behavior space
            anchor_policy, anchor_bd = archive.sample_uniform()

            # NOTE: target_bd is the target behavior we want to express now
            target_bd = tuple(min(bd + (np.random.normal() * target_bd_noise), 1.0) for bd in anchor_bd)

            mu = get_flat_params(anchor_policy).numpy().flatten()
            cov = np.ones(num_params) * sigma_init

        # Create a population of solutions from current Gaussian distribution
        for _ in range(population_size):
            # Sample using diagonal covariance: mu + sqrt(cov) * N(0, I) <- this is from CEM
            sample = mu + np.sqrt(cov) * np.random.randn(num_params)
            solutions.append(sample)

        # Evaluate half the population now
        cem_steps = []
        for i in range(num_cem):
            set_flat_params(actor, torch.from_numpy(solutions[i]).float())
            reward, steps = evaluate(actor)

            total_steps += steps
            rewards.append(reward)
            cem_steps.append(steps)

        # Number of critic training steps per agent: the mean episode length of the half just evaluated
        grad_steps = sum(cem_steps) // len(cem_steps)

        # The other half, first send to RL, then evaluate. (After the warm-up iterations, otherwise just do CEM-ES as a warm-up)
        if iteration >= warmup_iterations:

            # Train the critic once every critic_update_interval env steps
            if total_steps >= target_steps_for_critic:
                target_steps_for_critic += critic_update_interval
                train_critic = True
                num_grad_steps = grad_steps
            else:
                train_critic = False
                num_grad_steps = 0

            for i in range(num_cem, population_size):
                # load the weights of the other half of the population, then do td3
                set_flat_params(actor, torch.from_numpy(solutions[i]).float())
                policy = copy.deepcopy(actor).to(device)

                if train_critic:
                    td3.train_critic(policy, replay_buffers, num_grad_steps)

                refined = td3.refine(policy, target_bd, replay_buffers).cpu()

                # The last num_target_labeled refined policies label their transitions with the target descriptor,
                # the rest with the descriptor they actually expressed
                label = target_bd if i >= population_size - num_target_labeled else None
                fit, steps = evaluate(refined, descriptor=label)

                total_steps += steps
                rewards.append(fit)

                solutions[i] = get_flat_params(refined).numpy()  # each rl improved policy is put back in the population
        else:
            num_grad_steps = 0
            for i in range(num_cem, population_size):
                set_flat_params(actor, torch.from_numpy(solutions[i]).float())
                reward, steps = evaluate(actor)

                total_steps += steps
                rewards.append(reward)
            if iteration == warmup_iterations - 1:
                target_steps_for_critic = total_steps  # starting steps holds the starting count for env steps for RL

        # Update rule logic:
        scores = np.array(rewards)
        scores *= -1  # -1 so argsort sorts in descending
        idx_sorted = np.argsort(scores)  # sorted indices (ascending of negated -> descending original)

        old_mu = mu.copy()
        damp = damp * damp_tau + (1 - damp_tau) * damp_limit

        # Select top parents solutions to use in gradient calculation
        top_indices = idx_sorted[:parents]
        top_solutions = np.array(solutions)[top_indices]  # shape: (parents, num_params)

        # Update mean using weighted sum of top solutions
        mu = weights @ top_solutions

        # Calculate deviation of top solutions from old mean
        z = top_solutions - old_mu

        # Update diagonal covariance using weighted average of squared deviations
        cov = (1 / parents) * weights @ (z * z) + damp * np.ones(num_params)

        # Evaluate current mean policy + RL policies for printing purposes (not part of algo)
        set_flat_params(actor, torch.from_numpy(mu).float())
        mean_reward, _ = evaluate(actor, record=False)
        max_rl, min_rl = max(rewards[num_cem:]), min(rewards[num_cem:])

        qd_score = archive.qd_score()
        max_fitness = archive.max_fitness()

        log_line = (
            f"Iteration {iteration+1}, Step {total_steps}, Num Grad Steps: {num_grad_steps}, "
            f"Mean Actor Reward: {mean_reward:.2f}, Min RL: {min_rl:.2f}, Max RL: {max_rl:.2f}, "
            f"QD Score: {qd_score}, Max Fit: {max_fitness}, Archive Size: {len(archive)}"
        )
        print(log_line)
        with open(log_path, "a") as f:
            f.write(log_line + "\n")

        qd_scores.append((total_steps, qd_score))
        max_fitnesses.append((total_steps, max_fitness))
        archive_sizes.append((total_steps, len(archive)))

    return qd_scores, max_fitnesses, archive_sizes
