from typing import Tuple

import brax
import flax.linen as nn
import jax
import jax.numpy as jnp
from brax.envs import State as EnvState
from qdax.core.neuroevolution.buffers.buffer import QDTransition
from qdax.types import Params, RNGKey


def play_step_fn_skill_discovery(
    env_state: EnvState,
    policy_params: Params,
    random_key: RNGKey,
    env: brax.envs.Env,
    policy_network: nn.Module,
    deterministic: bool = True,
) -> Tuple[EnvState, Params, RNGKey, QDTransition]:
    """Plays a step in the environment. Concatenates skills to the observation
    vector, selects an action according to SAC rule and performs the environment
    step.
    """
    skills = policy_params["skills"]
    policy_params_ = policy_params["policy_params"]
    obs = jnp.concatenate([env_state.obs, skills[None]], axis=1)

    # If the env does not support state descriptor, we set it to (0,0)
    if "state_descriptor" in env_state.info:
        state_desc = env_state.info["state_descriptor"]
    else:
        state_desc = jnp.zeros((env_state.obs.shape[0], 2))

    dist_params = policy_network.apply(policy_params_, obs)
    actions = jax.nn.tanh(dist_params[..., : dist_params.shape[-1] // 2])

    next_env_state = env.step(env_state, actions)
    next_obs = jnp.concatenate([next_env_state.obs, skills[None]], axis=1)
    if "state_descriptor" in next_env_state.info:
        next_state_desc = next_env_state.info["state_descriptor"]
    else:
        next_state_desc = jnp.zeros((next_env_state.obs.shape[0], 2))
    truncations = next_env_state.info["truncation"]
    transition = QDTransition(
        obs=obs,
        next_obs=next_obs,
        state_desc=state_desc,
        next_state_desc=next_state_desc,
        rewards=next_env_state.reward,
        dones=next_env_state.done,
        actions=actions,
        truncations=truncations,
    )

    return next_env_state, policy_params, random_key, transition
