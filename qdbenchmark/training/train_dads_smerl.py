import logging
import os
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, List, Optional, Tuple

import brax
import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from hydra.core.config_store import ConfigStore
from jax.flatten_util import ravel_pytree
from qdax import environments
from qdax.baselines.dads_smerl import DADSSMERL, DadsSmerlConfig
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax.core.neuroevolution.buffers.buffer import QDTransition, ReplayBuffer
from qdax.core.neuroevolution.buffers.trajectory_buffer import TrajectoryBuffer
from qdax.core.neuroevolution.mdp_utils import TrainingState
from qdax.core.neuroevolution.sac_utils import do_iteration_fn, warmstart_buffer
from qdax.utils.metrics import CSVLogger, default_qd_metrics
from qdax.utils.plotting import plot_2d_map_elites_repertoire, plot_skills_trajectory
from qdbenchmark.utils.logging import LoggingConfig
from qdbenchmark.utils.metrics import log_accumulated_metrics


@dataclass
class ExperimentConfig:
    """Configuration for this experiment script"""

    env_name: str
    seed: int
    env_batch_size: int
    warmup_steps: int
    buffer_size: int
    num_steps: Optional[int]
    time_limit: Optional[int]

    # SAC config
    batch_size: int
    episode_length: int
    grad_updates_per_step: float
    tau: float
    normalize_observations: bool
    learning_rate: float
    alpha_init: float
    discount: float
    reward_scaling: float
    hidden_layer_sizes: Tuple[int, ...]
    fix_alpha: bool
    # DADS config
    num_skills: int
    dynamics_update_freq: int
    normalize_target: bool
    descriptor_full_state: bool
    # SMERL config
    diversity_reward_scale: float
    smerl_target: float
    smerl_margin: float

    # QD Metrics config
    num_centroids: int
    num_init_cvt_samples: int
    fitness_range: Tuple[float, ...]

    logging: LoggingConfig

    descriptors_range: Optional[Any] = None


@hydra.main(config_path="config/dads_smerl", config_name="pointmaze")
def train(config: ExperimentConfig) -> None:
    """Launches and monitors the training of the agent."""

    assert (
        config.logging.save_checkpoints_period % config.logging.log_period == 0
    ), f"{config.logging.save_checkpoints_period} not a multiple/\
       of {config.logging.log_period}"

    if config.logging.save_to_gcp is not None and config.logging.save_to_gcp:
        assert config.logging.bucket_name is not None

    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().handlers[0].setLevel(logging.INFO)
    logger = logging.getLogger(f"{__name__}")

    output_dir = os.getcwd()

    csv_logger = CSVLogger("metrics.csv", header=["metric_name", "step", "value"])

    _policy_dir = os.path.join(output_dir, "checkpoints", "policy")
    _img_dir = os.path.join(output_dir, "images")
    os.makedirs(_policy_dir, exist_ok=True)
    os.makedirs(_img_dir, exist_ok=True)

    # Initialize environments
    env_batch_size = config.env_batch_size

    if config.env_name == "anttrap":
        wrappers_kwargs = [
            {"minval": [0.0, -8.0], "maxval": [jnp.inf, 8.0]},
            {},
        ]

    elif config.env_name == "ant_omni":
        wrappers_kwargs = [
            {"minval": [-jnp.inf, -jnp.inf], "maxval": [jnp.inf, jnp.inf]},
            {},
        ]

    else:
        wrappers_kwargs = None

    env = environments.create(
        env_name=config.env_name,
        batch_size=env_batch_size,
        episode_length=config.episode_length,
        auto_reset=True,
        qdax_wrappers_kwargs=wrappers_kwargs,
    )

    eval_env = environments.create(
        env_name=config.env_name,
        batch_size=env_batch_size,
        episode_length=config.episode_length,
        auto_reset=True,
        eval_metrics=True,
        qdax_wrappers_kwargs=wrappers_kwargs,
    )

    random_key = jax.random.PRNGKey(config.seed)
    env_state = jax.jit(env.reset)(rng=random_key)
    eval_env_first_state = jax.jit(eval_env.reset)(rng=random_key)

    # Initialize buffer
    dummy_transition = QDTransition.init_dummy(
        observation_dim=env.observation_size + config.num_skills,
        action_dim=env.action_size,
        descriptor_dim=env.behavior_descriptor_length,
    )
    replay_buffer = TrajectoryBuffer.init(
        buffer_size=config.buffer_size,
        transition=dummy_transition,
        env_batch_size=config.env_batch_size,
        episode_length=config.episode_length,
    )

    if config.descriptor_full_state:
        descriptor_size = env.observation_size
    else:
        descriptor_size = env.behavior_descriptor_length

    dads_config = DadsSmerlConfig(
        # SAC config
        batch_size=config.batch_size,
        episode_length=config.episode_length,
        grad_updates_per_step=config.grad_updates_per_step,
        tau=config.tau,
        normalize_observations=config.normalize_observations,
        learning_rate=config.learning_rate,
        alpha_init=config.alpha_init,
        discount=config.discount,
        reward_scaling=config.reward_scaling,
        hidden_layer_sizes=config.hidden_layer_sizes,
        fix_alpha=config.fix_alpha,
        # DADS config
        num_skills=config.num_skills,
        omit_input_dynamics_dim=descriptor_size,
        dynamics_update_freq=config.dynamics_update_freq,
        normalize_target=config.normalize_target,
        descriptor_full_state=config.descriptor_full_state,
        # SMERL config
        diversity_reward_scale=config.diversity_reward_scale,
        smerl_target=config.smerl_target,
        smerl_margin=config.smerl_margin,
    )

    dads = DADSSMERL(
        config=dads_config,
        action_size=env.action_size,
        descriptor_size=descriptor_size,
    )
    random_key, subkey = jax.random.split(random_key)
    training_state = dads.init(
        random_key=subkey,
        action_size=env.action_size,
        observation_size=env.observation_size,
        descriptor_size=descriptor_size,
    )

    random_key, subkey = jax.random.split(random_key)

    num_env_per_skill = env_batch_size // config.num_skills
    skills = jnp.concatenate(
        [jnp.eye(config.num_skills)] * num_env_per_skill,
        axis=0,
    )
    # Make play_step* functions scannable by passing static args beforehand
    play_eval_step = partial(
        dads.play_step_fn,
        env=eval_env,
        deterministic=True,
        evaluation=True,
        skills=skills,
    )

    play_step = partial(dads.play_step_fn, env=env, deterministic=False, skills=skills)

    update = dads.update

    eval_policy = partial(
        dads.eval_policy_fn,
        play_step_fn=play_eval_step,
        eval_env_first_state=eval_env_first_state,
        env_batch_size=config.env_batch_size,
    )

    # warmstart the buffer
    logger.warning("\n -----  Initialization -----")
    logger.warning(f"Warming up the buffer for {config.warmup_steps} warmup steps")
    init_warmup_time = time.time()
    random_key, subkey = jax.random.split(random_key)

    replay_buffer, env_state, training_state = warmstart_buffer(
        replay_buffer=replay_buffer,
        training_state=training_state,
        env_state=env_state,
        num_warmstart_steps=config.warmup_steps,
        env_batch_size=env_batch_size,
        play_step_fn=play_step,
    )
    warmump_duration = time.time() - init_warmup_time
    logger.warning(f"Duration of warmup: {warmump_duration:.2f}s")

    # Compute the centroids
    logger.warning("--- Compute the CVT centroids ---")
    if config.descriptors_range is None:
        minval, maxval = env.behavior_descriptor_limits

    else:
        minval = config.descriptors_range[0]
        maxval = config.descriptors_range[1]

    centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=config.num_init_cvt_samples,
        num_centroids=config.num_centroids,
        minval=minval,
        maxval=maxval,
        random_key=random_key,
    )

    init_fitnesses, init_descriptors = (
        -jnp.inf * jnp.ones((1,)),
        jnp.zeros((1, env.behavior_descriptor_length)),
    )

    policy_params = jax.tree_map(
        lambda x: x[None],
        training_state.policy_params,
    )
    policy_params = dict(policy_params)
    policy_params["skills"] = jnp.zeros((1, config.num_skills))

    repertoire = MapElitesRepertoire.init(
        genotypes=policy_params,
        fitnesses=init_fitnesses,
        descriptors=init_descriptors,
        centroids=centroids,
    )

    reward_offset = environments.reward_offset[config.env_name]
    metrics_fn = partial(
        default_qd_metrics, qd_offset=reward_offset * config.episode_length
    )
    do_iteration = partial(
        do_iteration_fn,
        env_batch_size=env_batch_size,
        grad_updates_per_step=config.grad_updates_per_step,
        play_step_fn=play_step,
        update_fn=update,
    )

    @jax.jit
    def _scan_do_iteration(
        carry: Tuple[TrainingState, brax.envs.State, ReplayBuffer],
        unused_arg: Any,
    ) -> Tuple[Tuple[TrainingState, brax.envs.State, ReplayBuffer], Any]:
        (
            training_state,
            env_state,
            replay_buffer,
            metrics,
        ) = do_iteration(*carry)
        return (training_state, env_state, replay_buffer), metrics

    # Main loop
    iteration_log_period = config.logging.log_period // env_batch_size
    current_step = 0
    init_time = time.time()
    if config.time_limit is not None:
        num_loops = jnp.inf
    elif config.num_steps is not None:
        total_num_iterations = config.num_steps // env_batch_size
        num_loops = total_num_iterations // iteration_log_period + 1
    else:
        raise NotImplementedError(
            "Either a time limit or a number of iterations must\
    be specified"
        )

    iteration = 0
    while iteration < num_loops:
        if iteration > 0:
            # Training part
            init_loop_time = time.time()
            # Perform `iteration_log_period` training steps
            (training_state, env_state, replay_buffer), (metrics) = jax.lax.scan(
                _scan_do_iteration,
                (training_state, env_state, replay_buffer),
                (),
                length=iteration_log_period,
            )

            jax.tree_map(lambda x: x.block_until_ready(), metrics)
            loop_duration = time.time() - init_loop_time
            logger.warning("-" * 60)
            logger.warning(f"Iteration duration: {loop_duration:.2f}s")

        last_step = current_step
        current_step = iteration_log_period * iteration * env_batch_size

        # Evaluation
        # Policy evaluation
        true_return, true_returns, diversity_returns, state_desc = eval_policy(
            training_state=training_state
        )
        true_return.block_until_ready()

        # TODO: improve bd extraction
        if "uni" in config.env_name:
            behavior_descriptor = jnp.nanmean(state_desc, axis=0)

        else:
            behavior_descriptor = np.apply_along_axis(
                lambda x: x[jnp.where(~jnp.isnan(x))][-1], 0, state_desc
            )  # type: ignore

        policy_params = dict(training_state.policy_params).copy()
        policy_params = jax.tree_map(
            lambda x: jnp.repeat(x[None], len(skills), axis=0), policy_params
        )
        policy_params["skills"] = skills
        repertoire = repertoire.add(
            policy_params,
            behavior_descriptor,
            true_returns,
        )

        if env.behavior_descriptor_length == 2:
            fig, ax = plot_2d_map_elites_repertoire(
                centroids=centroids,
                repertoire_fitnesses=repertoire.fitnesses,
                minval=minval,
                maxval=maxval,
                repertoire_descriptors=repertoire.descriptors,
                ax=None,
                vmin=config.fitness_range[0],
                vmax=config.fitness_range[1],
            )
            fig.savefig(os.path.join(_img_dir, f"repertoire_{current_step}"))

            plt.close(fig)

            if "uni" not in config.env_name:
                fig, ax = plot_skills_trajectory(
                    trajectories=state_desc.T,
                    skills=skills,
                    min_values=minval,
                    max_values=maxval,
                )
                fig.savefig(os.path.join(_img_dir, f"skills_{current_step}"))
                plt.close(fig)

        if current_step % config.logging.save_checkpoints_period == 0:
            saveable_policy_params, _ = ravel_pytree(training_state.policy_params)
            saveable_normalization_state, _ = ravel_pytree(
                training_state.normalization_running_stats
            )
            saveable_dynamics_params, _ = ravel_pytree(training_state.dynamics_params)

            jnp.save(
                os.path.join(_policy_dir, f"policy-{current_step}.npy"),
                saveable_policy_params,
            )

            jnp.save(
                os.path.join(
                    _policy_dir,
                    f"normalization-{current_step}.npy",
                ),
                saveable_normalization_state,
            )
            jnp.save(
                os.path.join(
                    _policy_dir,
                    f"dynamics-{current_step}.npy",
                ),
                saveable_dynamics_params,
            )

        # Logging part
        if iteration > 0:
            # Subsample the number of logged metrics as it can take a while to
            # log every single point
            metrics = jax.tree_map(
                lambda x: jnp.nanmean(x, axis=1)
                .flatten()[:: config.logging.metrics_subsample]
                .block_until_ready(),
                metrics,
            )

            log_accumulated_metrics(
                metrics=metrics,
                metric_logger=csv_logger,
                current_step=current_step,
                last_step=last_step,
            )
        metric = {
            "metric_name": "eval_mean_return",
            "value": float(true_return),
            "step": int(current_step),
        }
        csv_logger.log(metric)

        metric = {
            "metric_name": "environment_steps",
            "value": int(current_step),
            "step": int(current_step),
        }
        csv_logger.log(metric)

        qd_metrics = metrics_fn(repertoire)
        for metric_name in ["qd_score", "coverage", "max_fitness"]:
            metric = {
                "metric_name": metric_name,
                "value": float(qd_metrics[metric_name]),
                "step": int(current_step),
            }

            csv_logger.log(metric)

        iteration += 1
        if config.time_limit is not None:
            if time.time() - init_time > config.time_limit:
                logger.warning("Exiting due to time limit")
                break


if __name__ == "__main__":
    root_logger = logging.getLogger("root")

    class CheckTypesFilter(logging.Filter):
        def filter(self, record):
            return "check_types" not in record.getMessage()

    root_logger.addFilter(CheckTypesFilter())
    cs = ConfigStore.instance()
    cs.store(name="validate_experiment_config", node=ExperimentConfig)
    train()
