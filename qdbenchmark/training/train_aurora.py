import functools
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import hydra
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from hydra.core.config_store import ConfigStore
from qdax.core.aurora import AURORA
from qdax.core.containers.mapelites_repertoire import (
    MapElitesRepertoire,
    compute_cvt_centroids,
)
from qdax.core.containers.unstructured_repertoire import UnstructuredRepertoire
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.core.neuroevolution.networks.networks import MLP
from qdax.environments.bd_extractors import get_aurora_bd
from qdax.tasks.brax_envs import (
    create_brax_scoring_fn,
    make_policy_network_play_step_fn_brax,
    scoring_aurora_function,
)
from qdax.utils import train_seq2seq
from qdax.utils.metrics import CSVLogger
from qdbenchmark import environments
from qdbenchmark.utils.logging import LoggingConfig
from qdbenchmark.utils.plotting import plot_2d_map_elites_repertoire


@dataclass
class ExperimentConfig:
    """Configuration from this experiment script"""

    # Env config
    seed: int
    env_name: str
    episode_length: int
    policy_hidden_layer_sizes: Tuple[int, ...]

    # ME config
    num_iterations: Optional[int]
    time_limit: Optional[int]

    # num_evaluations: int
    batch_size: int
    single_init_state: bool

    # Grid config
    num_centroids: int
    passive_num_centroids: int
    num_init_cvt_samples: int

    # Emitter config
    iso_sigma: float
    line_sigma: float
    crossover_percentage: float

    lstm_batch_size: int  # 128

    observation_option: str  # "no_sd"
    hidden_size: int
    l_value_init: float

    traj_sampling_freq: int
    max_observation_size: int
    prior_descriptor_dim: int
    fitness_range: Tuple[float, ...]

    logging: LoggingConfig


@hydra.main(config_path="config/aurora", config_name="pointmaze")
def train(config: ExperimentConfig) -> None:
    assert (config.time_limit is None) or (
        config.num_iterations is None
    ), "Either set a time limit or a number of iterations, but not both."

    assert (
        config.logging.save_checkpoints_period % config.logging.log_period == 0
    ), f"{config.logging.save_checkpoints_period} not a multiple/\
       of {config.logging.log_period}"

    # setup logging
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger().handlers[0].setLevel(logging.INFO)
    logger = logging.getLogger(f"{__name__}")

    cwd = os.getcwd()
    output_dir = cwd

    _last_metrics_dir = os.path.join(output_dir, "checkpoints", "last_metrics")
    _last_repertoire_dir = os.path.join(output_dir, "checkpoints", "last_grid")

    _repertoire_dir = os.path.join(output_dir, "checkpoints", "grid")
    _repertoire_img_dir = os.path.join(output_dir, "images", "me_grids")

    _timings_dir = os.path.join(output_dir, "timings")
    _init_state_dir = os.path.join(output_dir, "init_state")

    _passive_repertoire_dir = os.path.join(output_dir, "checkpoints", "passive_grid")
    _passive_repertoire_img_dir = os.path.join(output_dir, "images", "passive_grid")

    os.makedirs(_last_metrics_dir, exist_ok=True)
    os.makedirs(_last_repertoire_dir, exist_ok=True)

    os.makedirs(_repertoire_dir, exist_ok=True)
    os.makedirs(_repertoire_img_dir, exist_ok=True)

    os.makedirs(_timings_dir, exist_ok=True)
    os.makedirs(_init_state_dir, exist_ok=True)

    os.makedirs(_passive_repertoire_dir, exist_ok=True)
    os.makedirs(_passive_repertoire_img_dir, exist_ok=True)

    # Init environment
    env_name = config.env_name
    env = environments.create(env_name, episode_length=config.episode_length)

    csv_logger = CSVLogger("metrics.csv", header=["metric_name", "step", "value"])

    # print("Env observation dim: ", env.observation_size)
    # print("Env action dim: ", env.action_size)

    # Init a random key
    random_key = jax.random.PRNGKey(config.seed)

    # Init policy network
    policy_layer_sizes = config.policy_hidden_layer_sizes + (env.action_size,)
    policy_network = MLP(
        layer_sizes=policy_layer_sizes,
        # activation=flax.linen.swish,
        kernel_init=jax.nn.initializers.lecun_uniform(),
        final_activation=jnp.tanh,
    )

    # Init population of controllers
    random_key, subkey = jax.random.split(random_key)
    keys = jax.random.split(subkey, num=config.batch_size)
    fake_batch = jnp.zeros(shape=(config.batch_size, env.observation_size))
    init_variables = jax.vmap(policy_network.init)(keys, fake_batch)

    # Create the initial environment states
    random_key, init_state_subkey = jax.random.split(random_key)
    keys = jnp.repeat(
        jnp.expand_dims(init_state_subkey, axis=0), repeats=config.batch_size, axis=0
    )
    reset_fn = jax.jit(jax.vmap(env.reset))
    init_states = reset_fn(keys)
    # Save initial state
    # with open(os.path.join(_init_state_dir, "init_states.pkl"), "wb") as file_to_save:
    #     init_state = jax.tree_util.tree_map(lambda x: x[0], init_states)
    #     pickle.dump(init_state, file_to_save)

    # Prepare the scoring function
    bd_extraction_fn = functools.partial(
        get_aurora_bd,
        option=config.observation_option,
        hidden_size=config.hidden_size,
        traj_sampling_freq=config.traj_sampling_freq,
        max_observation_size=config.max_observation_size,
    )

    play_step_fn = make_policy_network_play_step_fn_brax(
        policy_network=policy_network, env=env
    )
    scoring_fn = functools.partial(
        scoring_aurora_function,
        init_states=init_states,
        episode_length=config.episode_length,
        play_step_fn=play_step_fn,
        behavior_descriptor_extractor=bd_extraction_fn,
    )

    # Define the usual scoring function
    random_key, subkey = jax.random.split(random_key)

    usual_bd_extraction_fn = environments.behavior_descriptor_extractor[config.env_name]
    usual_scoring_fn, random_key = create_brax_scoring_fn(
        env=env,
        policy_network=policy_network,
        bd_extraction_fn=usual_bd_extraction_fn,
        random_key=subkey,
        episode_length=config.episode_length,
        deterministic=True,
    )
    # Define emitter
    variation_fn = functools.partial(
        isoline_variation, iso_sigma=config.iso_sigma, line_sigma=config.line_sigma
    )
    mixing_emitter = MixingEmitter(
        mutation_fn=lambda x, y: (x, y),
        variation_fn=variation_fn,
        variation_percentage=1.0,
        batch_size=config.batch_size,
    )

    # Get minimum reward value to make sure qd_score are positive
    reward_offset = environments.reward_offset[env_name]

    # Define a metrics function
    def metrics_fn(repertoire: UnstructuredRepertoire) -> Dict:
        # Get metrics
        repertoire_empty = repertoire.fitnesses == -jnp.inf
        qd_score = jnp.sum(repertoire.fitnesses, where=~repertoire_empty)
        # Add offset for positive qd_score
        qd_score += (
            reward_offset * config.episode_length * jnp.sum(1.0 - repertoire_empty)
        )
        coverage = 100 * jnp.mean(1.0 - repertoire_empty)
        max_fitness = jnp.max(repertoire.fitnesses)

        return {"qd_score": qd_score, "max_fitness": max_fitness, "coverage": coverage}

    # Instantiate MAP-Elites
    aurora = AURORA(
        scoring_function=scoring_fn,
        emitter=mixing_emitter,
        metrics_function=metrics_fn,
    )

    # Compute the centroids
    logger.warning("--- Compute the CVT centroids ---")
    minval, maxval = env.behavior_descriptor_limits
    aurora_dims = config.hidden_size
    # minval, maxval = jnp.zeros(aurora_dims),jnp.ones(aurora_dims)
    init_time = time.time()

    centroids = jnp.zeros(shape=(config.num_centroids, aurora_dims))
    duration = time.time() - init_time
    logger.warning(f"--- Duration for CVT centroids computation : {duration:.2f}s")

    aurora_dims = config.hidden_size
    centroids = jnp.zeros(shape=(config.num_centroids, aurora_dims))

    @jax.jit
    def update_scan_fn(carry: Any, unused: Any) -> Any:
        """Scan the udpate function."""
        (
            repertoire,
            random_key,
            model_params,
            mean_observations,
            std_observations,
        ) = carry

        # update
        (
            repertoire,
            _,
            metrics,
            random_key,
        ) = aurora.update(
            repertoire,
            None,
            random_key,
            model_params,
            mean_observations,
            std_observations,
        )

        return (
            (repertoire, random_key, model_params, mean_observations, std_observations),
            metrics,
        )

    # Init algorithmNorma
    # AutoEncoder Params and INIT
    obs_dim = jnp.minimum(env.observation_size, config.max_observation_size)
    if config.observation_option == "full":
        observations_dims = (
            config.episode_length // config.traj_sampling_freq,
            obs_dim + config.prior_descriptor_dim,
        )
    elif config.observation_option == "no_sd":
        observations_dims = (
            config.episode_length // config.traj_sampling_freq,
            obs_dim,
        )
    elif config.observation_option == "only_sd":
        observations_dims = (
            config.episode_length // config.traj_sampling_freq,
            config.prior_descriptor_dim,
        )
    else:
        ValueError("The chosen option is not correct.")

    # define the seq2seq model
    model = train_seq2seq.get_model(
        observations_dims[-1], True, hidden_size=config.hidden_size
    )

    # init the model params
    random_key, subkey = jax.random.split(random_key)
    model_params = train_seq2seq.get_initial_params(
        model, subkey, (1, *observations_dims)
    )

    # print(jax.tree_map(lambda x: x.shape, model_params))

    # define arbitrary observation's mean/std
    mean_observations = jnp.zeros(observations_dims[-1])
    std_observations = jnp.ones(observations_dims[-1])

    # design aurora's schedule
    default_update_base = 10
    update_base = int(jnp.ceil(default_update_base / config.logging.log_period))
    schedules = jnp.cumsum(jnp.arange(update_base, 1000, update_base))
    # print("Schedules: ", schedules)

    model_params = train_seq2seq.get_initial_params(
        model, subkey, (1, observations_dims[0], observations_dims[-1])
    )

    # print(jax.tree_map(lambda x: x.shape, model_params))

    mean_observations = jnp.zeros(observations_dims[-1])

    std_observations = jnp.ones(observations_dims[-1])

    logger.warning("--- Algorithm initialisation ---")
    total_training_time = 0.0
    start_time = time.time()

    l_value_init = config.l_value_init
    # init step of the aurora algorithm
    repertoire, _, random_key = aurora.init(
        init_variables,
        centroids,
        random_key,
        model_params,
        mean_observations,
        std_observations,
        l_value_init,
    )

    algo_init_time = time.time() - start_time
    total_training_time += algo_init_time
    logger.warning("--- Initialised ---")
    logger.warning("--- Starting the algorithm main process ---")
    current_step_estimation = 0

    # initializing means and stds and AURORA
    random_key, subkey = jax.random.split(random_key)
    model_params, mean_observations, std_observations = train_seq2seq.lstm_ae_train(
        subkey,
        repertoire,
        model_params,
        0,
        hidden_size=config.hidden_size,
        batch_size=config.lstm_batch_size,
    )

    minval_2, maxval_2 = env.behavior_descriptor_limits
    passive_centroids, random_key = compute_cvt_centroids(
        num_descriptors=env.behavior_descriptor_length,
        num_init_cvt_samples=config.num_init_cvt_samples,
        num_centroids=config.passive_num_centroids,
        minval=minval_2,
        maxval=maxval_2,
        random_key=subkey,
    )

    # Create the passive map elites repertoire
    # Initialize grid with default values
    default_fitnesses = -jnp.inf * jnp.ones(shape=config.passive_num_centroids)
    default_genotypes = jax.tree_map(
        lambda x: jnp.zeros(shape=(config.passive_num_centroids,) + x.shape[1:]),
        init_variables,
    )
    default_descriptors = jnp.zeros(
        shape=(config.passive_num_centroids, passive_centroids.shape[-1])
    )

    # fake initial one
    passive_repertoire = MapElitesRepertoire.init(
        genotypes=default_genotypes,
        fitnesses=default_fitnesses,
        descriptors=default_descriptors,
        centroids=passive_centroids,
    )

    current_step_estimation = 0
    total_training_time = 0.0
    time_duration = 0.0

    if config.time_limit is not None:
        num_loop_iterations = jnp.inf
    elif config.num_iterations is not None:
        num_loop_iterations = config.num_iterations // config.logging.log_period + 1
    else:
        raise NotImplementedError(
            "Either a time limit or a number of iterations must\
             be specified"
        )

    # Main loop
    n_target = 1024

    previous_error = jnp.sum(repertoire.fitnesses != -jnp.inf) - n_target

    iteration = 1  # to be consistent with other exp scripts
    while iteration < num_loop_iterations:
        logger.warning(
            f"--- Iteration indice : {iteration} out of {num_loop_iterations-1} ---"
        )
        logger.warning(
            f"--- Iteration number : {iteration * config.logging.log_period}"
            f" out of {(num_loop_iterations-1) * config.logging.log_period}---"
        )
        start_time = time.time()
        (
            (repertoire, random_key, model_params, mean_observations, std_observations),
            metrics,
        ) = jax.lax.scan(
            update_scan_fn,
            (repertoire, random_key, model_params, mean_observations, std_observations),
            (),
            length=config.logging.log_period,
        )
        time_duration = time.time() - start_time  # time for log_freq iterations
        total_training_time += time_duration

        # update nb steps estimation
        current_step_estimation += (
            config.batch_size * config.episode_length * config.logging.log_period
        )

        # Autoencoder Steps and CVC
        # individuals_in_repo = jnp.sum(repertoire.fitnesses != -jnp.inf)

        if (iteration + 1) in schedules:
            random_key, subkey = jax.random.split(random_key)

            logger.warning("--- Retraining the model ---")

            (
                model_params,
                mean_observations,
                std_observations,
            ) = train_seq2seq.lstm_ae_train(
                subkey,
                repertoire,
                model_params,
                iteration,
                hidden_size=config.hidden_size,
            )
            # RE-ADDITION OF ALL THE NEW BEHAVIOURAL DESCRIPTORS WITH THE NEW AE

            normalized_observations = (
                repertoire.observations - mean_observations
            ) / std_observations
            new_descriptors = model.apply(
                {"params": model_params}, normalized_observations, method=model.encode
            )
            repertoire = repertoire.init(
                genotypes=repertoire.genotypes,
                centroids=repertoire.centroids,
                fitnesses=repertoire.fitnesses,
                descriptors=new_descriptors,
                observations=repertoire.observations,
                l_value=repertoire.l_value,
            )
            num_indivs = jnp.sum(repertoire.fitnesses != -jnp.inf)

        elif iteration % 2 == 0:
            num_indivs = jnp.sum(repertoire.fitnesses != -jnp.inf)

            # l_value =  repertoire.l_value * (1+1*10e-7*(num_indivs-n_target))
            current_error = num_indivs - n_target
            change_rate = current_error - previous_error
            prop_gain = 1 * 10e-6
            l_value = (
                repertoire.l_value
                + (prop_gain * (current_error))
                + (prop_gain * change_rate)
            )
            previous_error = current_error
            # CVC Implementation to keep a Constant number of individuals in the Archive
            repertoire = repertoire.init(
                genotypes=repertoire.genotypes,
                centroids=repertoire.centroids,
                fitnesses=repertoire.fitnesses,
                descriptors=repertoire.descriptors,
                observations=repertoire.observations,
                l_value=l_value,
            )

        # Classic descriptor evaluation
        eval_fitnesses, eval_descriptors, eval_extra_scores, _ = usual_scoring_fn(
            repertoire.genotypes, random_key=random_key
        )

        # print("Eval fitnesses: ", eval_fitnesses)
        # print("Eval descriptors: ", eval_descriptors)

        passive_repertoire = passive_repertoire.add(
            repertoire.genotypes,
            eval_descriptors,
            eval_fitnesses,
        )

        passive_metrics = metrics_fn(passive_repertoire)

        # # Let's log

        # logger.warning(f"--- Current l-value: {repertoire.l_value:.5f}")
        # logger.warning(
        #     f"--- Current Number of Individuals: {metrics['num_indivs'][-1]:.2f}"
        # )
        # # Log data at current iterations
        # logger.warning(
        #     f"--- Current nb steps in env (estimation): {current_step_estimation:.2f}"
        # )
        # logger.warning(f"--- Time duration (batch of iteration): {time_duration:.2f}")
        # logger.warning(
        #     f"--- Time duration (Sum of iterations): {total_training_time:.2f}"
        # )
        # logger.warning(f"--- Current QD Score: {metrics['qd_score'][-1]:.2f}")
        # logger.warning(f"--- Current Coverage: {metrics['coverage'][-1]:.2f}%")
        # logger.warning(f"--- Current Max Fitness: {metrics['max_fitness'][-1]}")

        # logger.warning(
        #     f"--- Current Passive QD Score: {passive_metrics['qd_score']:.2f}"
        # )
        # logger.warning(
        #     f"--- Current Passive Coverage: {passive_metrics['coverage']:.2f}%"
        # )
        # logger.warning(
        #     f"--- Current Passive Max Fitness: {passive_metrics['max_fitness']}"
        # )

        if env.behavior_descriptor_length == 2:
            fig, ax = plot_2d_map_elites_repertoire(
                centroids=centroids,
                repertoire_fitnesses=eval_fitnesses,
                minval=minval,
                maxval=maxval,
                use_centroids=False,
                repertoire_descriptors=eval_descriptors,
                ax=None,
            )

            fig.savefig(
                os.path.join(
                    _repertoire_img_dir, f"repertoire_{current_step_estimation}"
                )
            )
            plt.close(fig)

            fig, ax = plot_2d_map_elites_repertoire(
                centroids=centroids,
                repertoire_fitnesses=repertoire.fitnesses,
                minval=jnp.nanmin(repertoire.descriptors[..., :2], axis=0),
                maxval=jnp.nanmax(repertoire.descriptors[..., :2], axis=0),
                use_centroids=False,
                repertoire_descriptors=repertoire.descriptors[..., :2],
                ax=None,
            )
            fig.savefig(
                os.path.join(
                    _repertoire_img_dir, f"aurora_latent_{current_step_estimation}"
                )
            )
            plt.close(fig)

            # plot the passive repertoire

            fig, ax = plot_2d_map_elites_repertoire(
                centroids=passive_centroids,
                repertoire_fitnesses=passive_repertoire.fitnesses,
                minval=minval_2,
                maxval=maxval_2,
                repertoire_descriptors=passive_repertoire.descriptors,
                ax=None,
                vmin=config.fitness_range[0],
                vmax=config.fitness_range[1],
            )
            fig.savefig(
                os.path.join(
                    _passive_repertoire_img_dir,
                    f"passive_repertoire_{current_step_estimation}",
                )
            )

            plt.close(fig)

        # Store the latest controllers of the repertoire
        repertoire.save(path=_last_repertoire_dir + "/")

        # store the latest controllers
        store_entire_me_grid = True
        if store_entire_me_grid:
            repertoire.save(path=os.path.join(_repertoire_dir))
            passive_repertoire.save(path=os.path.join(_passive_repertoire_dir))

        # Logging part
        logger.warning("Start logging metrics")

        metrics = {
            "num_steps": float(current_step_estimation),
            "batched_iteration_duration": float(time_duration),
            "aurora qd_score": metrics["qd_score"][-1],
            "aurora coverage": metrics["coverage"][-1],
            "aurora max_fitness": metrics["max_fitness"][-1],
            "qd_score": passive_metrics["qd_score"],
            "coverage": passive_metrics["coverage"],
            "max_fitness": passive_metrics["max_fitness"],
        }

        for name, value in metrics.items():
            metric = {
                "metric_name": name,
                "value": float(value),
                "step": int(current_step_estimation),
            }
            csv_logger.log(metric)

        logger.warning("End logging metrics")

        iteration += 1
        if config.time_limit is not None:
            if time.time() - init_time > config.time_limit:
                logger.warning("Exiting due to time limit")
                break

    duration = time.time() - init_time
    # Save final plot
    if env.behavior_descriptor_length == 2:
        fig, ax = plot_2d_map_elites_repertoire(
            centroids=centroids,
            repertoire_fitnesses=eval_fitnesses,
            minval=jnp.nanmin(eval_descriptors, axis=0),
            maxval=jnp.nanmax(eval_descriptors, axis=0),
            use_centroids=False,
            repertoire_descriptors=eval_descriptors,
            ax=None,
        )

        fig.savefig(
            os.path.join(_repertoire_img_dir, f"repertoire_{current_step_estimation}")
        )
        plt.close(fig)
    # Save final repertoire
    repertoire.save(path=_last_repertoire_dir + "/")


if __name__ == "__main__":
    cs = ConfigStore.instance()
    cs.store(name="validate_experiment_config", node=ExperimentConfig)
    train()
