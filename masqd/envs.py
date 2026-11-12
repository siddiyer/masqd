import mujoco
from gymnasium_robotics import mamujoco_v1

from .rollout import FLOOR_GEOM_ID


def make_env(env_config):
    """Creates the MaMuJoCo parallel env described by the env section of a config."""
    env = mamujoco_v1.parallel_env(env_config["scenario"], env_config["agent_conf"])

    # Make sure the descriptor geom IDs refer to the floor and feet we expect
    model = env.single_agent_env.unwrapped.model
    expected = [(FLOOR_GEOM_ID, "floor")] + list(zip(env_config["foot_geom_ids"], env_config["foot_geom_names"]))
    for geom_id, name in expected:
        actual = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if actual != name:
            raise ValueError(f"{env_config['name']}: geom {geom_id} is '{actual}', expected '{name}'")

    return env
