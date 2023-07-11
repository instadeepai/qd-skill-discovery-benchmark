import functools
from typing import Any, Callable, Dict, List, Optional, Union

import brax
import jax.numpy as jnp
from qdax.environments.base_wrappers import QDEnv, StateDescriptorResetWrapper
from qdax.environments.bd_extractors import (
    get_feet_contact_proportion,
    get_final_xy_position,
)
from qdax.environments.locomotion_wrappers import (
    FeetContactWrapper,
    NoForwardRewardWrapper,
    XYPositionWrapper,
)
from qdax.environments.pointmaze import PointMaze

from qdbenchmark.environments.actuator_wrappers import ActuatorStrengthWrapper
from qdbenchmark.environments.default_position_wrapper import DefaultPositionWrapper
from qdbenchmark.environments.exploration_wrappers import MazeWrapper, TrapWrapper
from qdbenchmark.environments.hurdles_wrapper import HurdlesWrapper
from qdbenchmark.environments.physics_wrappers import FrictionWrapper, GravityWrapper

# experimentally determined offset (except for antmaze)
# should be enough to have only positive rewards but no guarantee
reward_offset = {
    "pointmaze": 2.3431,
    "anttrap": 3.38,
    "antmaze": 40.32,
    "ant_omni": 3.0,
    "ant_uni": 3.24,
    "halfcheetah_uni": 9.231,
    "walker2d_uni": 1.413,
}

behavior_descriptor_extractor = {
    "pointmaze": get_final_xy_position,
    "anttrap": get_final_xy_position,
    "antmaze": get_final_xy_position,
    "ant_omni": get_final_xy_position,
    "ant_uni": get_feet_contact_proportion,
    "halfcheetah_uni": get_feet_contact_proportion,
    "walker2d_uni": get_feet_contact_proportion,
}

_qdbenchmark_envs = {
    "pointmaze": PointMaze,
}

_qdbenchmark_custom_envs = {
    "anttrap": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, TrapWrapper],
        "kwargs": [{"minval": [0.0, -8.0], "maxval": [30.0, 8.0]}, {}],
    },
    "antmaze": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, MazeWrapper],
        "kwargs": [{"minval": [-5.0, -5.0], "maxval": [40.0, 40.0]}, {}],
    },
    "ant_omni": {
        "env": "ant",
        "wrappers": [XYPositionWrapper, NoForwardRewardWrapper],
        "kwargs": [{"minval": [-30.0, -30.0], "maxval": [30.0, 30.0]}, {}],
    },
    "ant_uni": {"env": "ant", "wrappers": [FeetContactWrapper], "kwargs": [{}, {}]},
    "halfcheetah_uni": {
        "env": "halfcheetah",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
    },
    "walker2d_uni": {
        "env": "walker2d",
        "wrappers": [FeetContactWrapper],
        "kwargs": [{}],
    },
    "halfcheetah_hurdles": {
        "env": "halfcheetah",
        "wrappers": [HurdlesWrapper],
        "kwargs": [{}],
    },
}


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    eval_metrics: bool = False,
    gravity_multiplier: Optional[float] = None,
    friction_multiplier: Optional[float] = None,
    actuator_update: Optional[Dict[str, float]] = None,
    default_position_update: Optional[Dict[str, jnp.ndarray]] = None,
    qdax_wrappers_kwargs: Optional[List] = None,
    **kwargs: Any,
) -> Union[brax.envs.Env, QDEnv]:
    """Creates an Env with a specified brax system.
    Please use namespace to avoid confusion between this function and
    brax.envs.create.
    """

    if env_name in brax.envs._envs.keys():
        env = brax.envs._envs[env_name](legacy_spring=True, **kwargs)
    elif env_name in _qdbenchmark_envs.keys():
        env = _qdbenchmark_envs[env_name](**kwargs)
    elif env_name in _qdbenchmark_custom_envs.keys():
        base_env_name = _qdbenchmark_custom_envs[env_name]["env"]
        if base_env_name in brax.envs._envs.keys():
            env = brax.envs._envs[base_env_name](legacy_spring=True, **kwargs)
        elif base_env_name in _qdbenchmark_envs.keys():
            # WARNING: not general!! temporary trick
            env = _qdbenchmark_envs[base_env_name](**kwargs)  # type: ignore

        # roll with qdbenchmark wrappers
        wrappers = _qdbenchmark_custom_envs[env_name]["wrappers"]
        if qdax_wrappers_kwargs is None:
            kwargs_list = _qdbenchmark_custom_envs[env_name]["kwargs"]
        else:
            kwargs_list = qdax_wrappers_kwargs

        for wrapper, kwargs in zip(wrappers, kwargs_list):  # type: ignore
            env = wrapper(env, base_env_name, **kwargs)  # type: ignore
    else:
        raise NotImplementedError("This environment name does not exist!")

    # custom wrapper
    if gravity_multiplier is not None:
        env = GravityWrapper(env, gravity_multiplier)
    if friction_multiplier is not None:
        env = FrictionWrapper(env, friction_multiplier)
    if actuator_update is not None:
        for actuator_name, strength_multiplier in actuator_update.items():
            env = ActuatorStrengthWrapper(
                env,
                actuator_name=actuator_name,
                strength_multiplier=strength_multiplier,
            )
    if default_position_update is not None:
        for body_name, body_position in default_position_update.items():
            env = DefaultPositionWrapper(
                env,
                body_name=body_name,
                body_position=body_position,
            )

    # brax wrappers
    if episode_length is not None:
        env = brax.envs.wrappers.EpisodeWrapper(env, episode_length, action_repeat)
    if batch_size:
        env = brax.envs.wrappers.VectorWrapper(env, batch_size)
    if auto_reset:
        env = brax.envs.wrappers.AutoResetWrapper(env)
        if env_name in _qdbenchmark_custom_envs.keys():
            env = StateDescriptorResetWrapper(env)
    if eval_metrics:
        env = brax.envs.wrappers.EvalWrapper(env)
    return env


def create_fn(env_name: str, **kwargs: Any) -> Callable[..., brax.envs.Env]:
    """Returns a function that when called, creates an Env.
    Please use namespace to avoid confusion between this function and
    brax.envs.create_fn.
    """
    return functools.partial(create, env_name, **kwargs)
