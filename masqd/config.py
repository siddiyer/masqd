import yaml

# Expected type of every setting, per config section
SCHEMA = {
    "env": {
        "name": str,
        "scenario": str,
        "agent_conf": str,
        "num_agents": int,
        "state_dim": int,
        "action_dim": int,
        "foot_geom_ids": list,
        "foot_geom_names": list,
        "bd_decimals": int,
    },
    "masqd": {
        "max_env_steps": int,
        "population_size": int,
        "elite_frac": float,
        "warmup_iterations": int,
        "resample_every": int,
        "sigma_init": float,
        "damp_init": float,
        "damp_limit": float,
        "damp_tau": float,
        "target_bd_noise": float,
        "critic_update_interval": int,
        "num_target_labeled": int,
    },
    "networks": {
        "actor_hidden_size": int,
        "critic_hidden_size": int,
    },
    "dc_td3": {
        "max_action": float,
        "discount": float,
        "tau": float,
        "policy_noise": float,
        "noise_clip": float,
        "policy_freq": int,
        "lr": float,
        "batch_size": int,
        "similarity_scale": float,
        "refine_steps": int,
        "refine_lr": float,
        "refine_batch_size": int,
    },
    "replay_buffer": {
        "max_size": int,
    },
}


def load_config(path):
    """Loads a YAML config and checks that it has exactly the sections, keys and value types in SCHEMA."""
    with open(path) as f:
        config = yaml.safe_load(f)

    if set(config) != set(SCHEMA):
        raise ValueError(f"{path}: expected sections {sorted(SCHEMA)}, got {sorted(config)}")

    for section, fields in SCHEMA.items():
        values = config[section]
        missing = sorted(set(fields) - set(values))
        unknown = sorted(set(values) - set(fields))
        if missing:
            raise ValueError(f"{path}: section '{section}' is missing {missing}")
        if unknown:
            raise ValueError(f"{path}: section '{section}' has unknown keys {unknown}")

        for key, expected in fields.items():
            value = values[key]
            if expected is float and isinstance(value, int) and not isinstance(value, bool):
                values[key] = float(value)
            elif not isinstance(value, expected) or isinstance(value, bool):
                hint = " (YAML reads numbers like 3e-4 as text; write 3.0e-4 or 0.0003)" if expected is float and isinstance(value, str) else ""
                raise TypeError(f"{path}: {section}.{key} should be of type {expected.__name__}, got {value!r}{hint}")

    env = config["env"]
    if not env["foot_geom_ids"] or len(env["foot_geom_ids"]) != len(env["foot_geom_names"]):
        raise ValueError(f"{path}: env.foot_geom_ids and env.foot_geom_names must be non-empty and the same length")

    return config
