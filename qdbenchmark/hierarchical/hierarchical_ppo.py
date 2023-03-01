# Copyright 2022 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Proximal policy optimization training from Brax adapted for hierarchical RL.

See: https://arxiv.org/pdf/1707.06347.pdf
"""


import functools
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

import flax
import jax
import jax.numpy as jnp
import optax
from absl import logging
from qdbenchmark.hierarchical.utils import Categorical

from brax import envs
from brax.io import model
from brax.training import distribution, networks, normalization, pmap
from brax.training.ppo import StepData, TrainingState, compute_gae
from brax.training.types import Params, PRNGKey


def compute_ppo_loss(
    models: Dict[str, Params],
    data: StepData,
    rng: jnp.ndarray,
    parametric_action_distribution: distribution.ParametricDistribution,
    policy_apply: Any,
    value_apply: Any,
    entropy_cost: float = 1e-4,
    discounting: float = 0.9,
    reward_scaling: float = 1.0,
    lambda_: float = 0.95,
    ppo_epsilon: float = 0.3,
):
    """Computes PPO loss."""
    policy_params, value_params = models["policy"], models["value"]
    policy_logits = policy_apply(policy_params, data.obs[:-1])
    baseline = value_apply(value_params, data.obs)
    baseline = jnp.squeeze(baseline, axis=-1)

    # Use last baseline value (from the value function) to bootstrap.
    bootstrap_value = baseline[-1]
    baseline = baseline[:-1]

    # At this point, we have unroll length + 1 steps. The last step is only used
    # as bootstrap value, so it's removed.

    # already removed at data generation time
    # actions = actions[:-1]
    # logits = logits[:-1]

    rewards = data.rewards[1:] * reward_scaling
    truncation = data.truncation[1:]
    termination = data.dones[1:] * (1 - truncation)

    target_action_log_probs = parametric_action_distribution.log_prob(
        policy_logits, data.actions
    )
    behaviour_action_log_probs = parametric_action_distribution.log_prob(
        data.logits, data.actions
    )

    vs, advantages = compute_gae(
        truncation=truncation,
        termination=termination,
        rewards=rewards,
        values=baseline,
        bootstrap_value=bootstrap_value,
        lambda_=lambda_,
        discount=discounting,
    )
    rho_s = jnp.exp(target_action_log_probs - behaviour_action_log_probs)
    surrogate_loss1 = rho_s * advantages
    surrogate_loss2 = jnp.clip(rho_s, 1 - ppo_epsilon, 1 + ppo_epsilon) * advantages

    policy_loss = -jnp.mean(jnp.minimum(surrogate_loss1, surrogate_loss2))

    # Value function loss
    v_error = vs - baseline
    v_loss = jnp.mean(v_error * v_error) * 0.5 * 0.5

    # Entropy reward
    entropy = jnp.mean(parametric_action_distribution.entropy(policy_logits, rng))
    entropy_loss = entropy_cost * -entropy

    return policy_loss + v_loss + entropy_loss, {
        "total_loss": policy_loss + v_loss + entropy_loss,
        "policy_loss": policy_loss,
        "v_loss": v_loss,
        "entropy_loss": entropy_loss,
        "policy_logits": policy_logits.mean(),
        "obs_mean": data.obs.mean(),
    }


def train(
    environment_fn: Callable[..., envs.Env],
    num_timesteps,
    num_h_steps: int,
    episode_length: int,
    num_skills: int,
    skill_inference_fn: Callable,
    action_repeat: int = 1,
    num_envs: int = 1,
    max_devices_per_host: Optional[int] = None,
    num_eval_envs: int = 128,
    learning_rate=1e-4,
    entropy_cost=1e-4,
    discounting=0.9,
    seed=0,
    unroll_length=10,
    batch_size=32,
    num_minibatches=16,
    num_update_epochs=2,
    log_frequency=10,
    normalize_observations=False,
    reward_scaling=1.0,
    progress_fn: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    checkpoint_dir: Optional[str] = None,
):
    """PPO training."""
    assert batch_size * num_minibatches % num_envs == 0
    xt = time.time()

    process_count = jax.process_count()
    process_id = jax.process_index()
    local_device_count = jax.local_device_count()
    local_devices_to_use = local_device_count
    if max_devices_per_host:
        local_devices_to_use = min(local_devices_to_use, max_devices_per_host)
    logging.info(
        "Device count: %d, process count: %d (id %d), local device count: %d, "
        "devices to be used count: %d",
        jax.device_count(),
        process_count,
        process_id,
        local_device_count,
        local_devices_to_use,
    )

    key = jax.random.PRNGKey(seed)
    key, key_models, key_env, key_eval = jax.random.split(key, 4)
    # Make sure every process gets a different random key, otherwise they will be
    # doing identical work.
    key_env = jax.random.split(key_env, process_count)[process_id]
    key = jax.random.split(key, process_count)[process_id]
    # key_models should be the same, so that models are initialized the same way
    # for different processes

    core_env = environment_fn(
        action_repeat=action_repeat,
        batch_size=num_envs // local_devices_to_use // process_count,
        episode_length=episode_length,
    )
    key_envs = jax.random.split(key_env, local_devices_to_use)
    step_fn = jax.jit(core_env.step)
    reset_fn = jax.jit(jax.vmap(core_env.reset))
    first_state = reset_fn(key_envs)

    eval_env = environment_fn(
        action_repeat=action_repeat,
        batch_size=num_eval_envs,
        episode_length=episode_length,
        eval_metrics=True,
    )
    eval_step_fn = jax.jit(eval_env.step)
    eval_first_state = jax.jit(eval_env.reset)(key_eval)

    parametric_action_distribution = distribution.NormalTanhDistribution(
        event_size=core_env.action_size
    )

    h_parametric_action_distribution = Categorical(event_size=1)

    # h_policy_model, h_value_model = networks.make_models(
    #     parametric_action_distribution.param_size, core_env.observation_size
    # )
    h_policy_model = networks.make_model(
        [
            32,
            32,
            32,
            32,
            num_skills,
        ],
        core_env.observation_size,
        activation=jax.nn.relu,
    )
    h_value_model = networks.make_model(
        [256, 256, 256, 256, 256, 1],
        core_env.observation_size,
        activation=jax.nn.relu,
    )

    key_policy, key_value = jax.random.split(key_models)

    optimizer = optax.adam(learning_rate=learning_rate)
    init_params = {
        "policy": h_policy_model.init(key_policy),
        "value": h_value_model.init(key_value),
    }
    optimizer_state = optimizer.init(init_params)
    optimizer_state, init_params = pmap.bcast_local_devices(
        (optimizer_state, init_params), local_devices_to_use
    )

    (
        normalizer_params,
        obs_normalizer_update_fn,
        obs_normalizer_apply_fn,
    ) = normalization.create_observation_normalizer(
        core_env.observation_size,
        normalize_observations,
        num_leading_batch_dims=2,
        pmap_to_devices=local_devices_to_use,
    )

    key_debug = jax.random.PRNGKey(seed + 666)

    loss_fn = functools.partial(
        compute_ppo_loss,
        parametric_action_distribution=h_parametric_action_distribution,
        policy_apply=h_policy_model.apply,
        value_apply=h_value_model.apply,
        entropy_cost=entropy_cost,
        discounting=discounting,
        reward_scaling=reward_scaling,
    )

    grad_loss = jax.grad(loss_fn, has_aux=True)

    @jax.jit
    def run_eval(
        state, key, policy_params, normalizer_params
    ) -> Tuple[envs.State, PRNGKey]:
        policy_params = jax.tree_map(lambda x: x[0], policy_params)

        (state, _, _, key), (rewards, dones) = jax.lax.scan(
            do_one_h_step_eval,
            (state, policy_params, normalizer_params, key),
            (),
            length=episode_length // action_repeat // num_h_steps,
        )
        return state, rewards, dones, key

    def do_one_h_step_eval(carry, unused_target_t):
        state, h_policy_params, normalizer_params, key = carry
        key, key_sample = jax.random.split(key)

        obs = obs_normalizer_apply_fn(
            jax.tree_map(lambda x: x[0], normalizer_params), state.obs
        )

        logits = h_policy_model.apply(h_policy_params, obs)
        actions = h_parametric_action_distribution.sample(logits, key_sample)
        key, key_generate_unroll = jax.random.split(key)

        (nstate, _, key), (rewards, dones) = jax.lax.scan(
            do_one_step_eval,
            (state, actions, key),
            (),
            length=num_h_steps,
        )
        rewards = jnp.concatenate([rewards, jnp.expand_dims(nstate.reward, axis=0)])
        dones = jnp.concatenate([dones, jnp.expand_dims(nstate.done, axis=0)])

        return (nstate, h_policy_params, normalizer_params, key), (
            rewards,
            dones,
        )

    def do_one_step_eval(carry, unused_target_t):
        state, skill, key = carry
        key, key_sample = jax.random.split(key)

        actions = skill_inference_fn(
            state.obs, random_key=key_sample, skill=skill, deterministic=True
        )
        nstate = eval_step_fn(state, actions)
        return (nstate, skill, key), (state.reward, state.done)

    def generate_h_unroll(carry, unused_target_t):
        state, normalizer_params, h_policy_params, key = carry
        (state, _, _, key), data = jax.lax.scan(
            do_one_h_step,
            (state, normalizer_params, h_policy_params, key),
            (),
            length=unroll_length,
        )
        data = data.replace(
            obs=jnp.concatenate([data.obs, jnp.expand_dims(state.obs, axis=0)]),
            rewards=jnp.concatenate(
                [data.rewards, jnp.expand_dims(state.reward, axis=0)]
            ),
            dones=jnp.concatenate([data.dones, jnp.expand_dims(state.done, axis=0)]),
            truncation=jnp.concatenate(
                [data.truncation, jnp.expand_dims(state.info["truncation"], axis=0)]
            ),
        )
        return (state, normalizer_params, h_policy_params, key), data

    def do_one_h_step(carry, unused_target_t):
        state, normalizer_params, h_policy_params, key = carry
        key, key_sample = jax.random.split(key)
        obs = obs_normalizer_apply_fn(normalizer_params, state.obs)
        # obs = state.obs
        logits = h_policy_model.apply(h_policy_params, obs)
        actions = h_parametric_action_distribution.sample_no_postprocessing(
            logits, key_sample
        )
        # postprocessed_actions = h_parametric_action_distribution.postprocess(actions)
        key, key_generate_unroll = jax.random.split(key)

        (nstate, _, _), data = generate_unroll(
            (
                state,
                actions,
                key_generate_unroll,
            ),
            None,
        )

        print("rewards shape:", data.rewards.shape)
        rewards = data.rewards
        dones = data.dones

        is_done = jnp.clip(jnp.cumsum(dones, axis=0), 0, 1)
        mask = jnp.roll(is_done, 1, axis=0)
        mask = mask.at[0, :].set(0)

        rewards = jnp.sum(rewards * (1.0 - mask), axis=0)
        dones = jnp.clip(jnp.sum(dones, axis=0), 0, 1)
        truncations = jnp.clip(jnp.sum(data.truncation, axis=0), 0, 1)
        return (nstate, normalizer_params, h_policy_params, key), StepData(
            obs=state.obs,
            rewards=rewards,
            dones=dones,
            truncation=truncations,
            actions=actions,
            logits=logits,
        )

    def generate_unroll(carry, unused_target_t):
        state, skill, key = carry
        (state, _, key), data = jax.lax.scan(
            do_one_step,
            (state, skill, key),
            (),
            length=num_h_steps,
        )
        data = data.replace(
            obs=jnp.concatenate([data.obs, jnp.expand_dims(state.obs, axis=0)]),
            rewards=jnp.concatenate(
                [data.rewards, jnp.expand_dims(state.reward, axis=0)]
            ),
            dones=jnp.concatenate([data.dones, jnp.expand_dims(state.done, axis=0)]),
            truncation=jnp.concatenate(
                [data.truncation, jnp.expand_dims(state.info["truncation"], axis=0)]
            ),
        )
        return (state, skill, key), data

    def do_one_step(carry, unused_target_t):
        state, skill, key = carry
        key, key_sample = jax.random.split(key)

        actions = skill_inference_fn(
            state.obs, random_key=key_sample, skill=skill, deterministic=False
        )
        nstate = step_fn(state, actions)
        return (nstate, skill, key), StepData(
            obs=state.obs,
            rewards=state.reward,
            dones=state.done,
            truncation=state.info["truncation"],
            actions=actions,
            logits=jnp.zeros_like(actions),
        )

    def update_model(carry, data):
        optimizer_state, params, key = carry
        key, key_loss = jax.random.split(key)
        loss_grad, metrics = grad_loss(params, data, key_loss)
        loss_grad = jax.lax.pmean(loss_grad, axis_name="i")

        params_update, optimizer_state = optimizer.update(loss_grad, optimizer_state)
        params = optax.apply_updates(params, params_update)

        return (optimizer_state, params, key), metrics

    def minimize_epoch(carry, unused_t):
        optimizer_state, params, data, key = carry
        key, key_perm, key_grad = jax.random.split(key, 3)
        permutation = jax.random.permutation(key_perm, data.obs.shape[1])

        def convert_data(data, permutation):
            data = jnp.take(data, permutation, axis=1, mode="clip")
            data = jnp.reshape(
                data, [data.shape[0], num_minibatches, -1] + list(data.shape[2:])
            )
            data = jnp.swapaxes(data, 0, 1)
            return data

        ndata = jax.tree_map(lambda x: convert_data(x, permutation), data)
        (optimizer_state, params, _), metrics = jax.lax.scan(
            update_model,
            (optimizer_state, params, key_grad),
            ndata,
            length=num_minibatches,
        )
        return (optimizer_state, params, data, key), metrics

    def run_epoch(carry: Tuple[TrainingState, envs.State], unused_t):
        training_state, state = carry
        key_minimize, key_generate_unroll, new_key = jax.random.split(
            training_state.key, 3
        )
        (state, _, _, _), data = jax.lax.scan(
            generate_h_unroll,
            (
                state,
                training_state.normalizer_params,
                training_state.params["policy"],
                key_generate_unroll,
            ),
            (),
            length=batch_size * num_minibatches // num_envs,
        )

        # make unroll first
        data = jax.tree_map(lambda x: jnp.swapaxes(x, 0, 1), data)
        data = jax.tree_map(
            lambda x: jnp.reshape(x, [x.shape[0], -1] + list(x.shape[3:])), data
        )

        # Update normalization params and normalize observations.
        normalizer_params = obs_normalizer_update_fn(
            training_state.normalizer_params, data.obs[:-1]
        )
        data = data.replace(obs=obs_normalizer_apply_fn(normalizer_params, data.obs))

        (optimizer_state, params, _, _), metrics = jax.lax.scan(
            minimize_epoch,
            (training_state.optimizer_state, training_state.params, data, key_minimize),
            (),
            length=num_update_epochs,
        )

        new_training_state = TrainingState(  # type: ignore
            optimizer_state=optimizer_state,
            params=params,
            normalizer_params=normalizer_params,
            key=new_key,
        )
        return (new_training_state, state), metrics

    num_epochs = num_timesteps // (
        batch_size * unroll_length * num_minibatches * action_repeat * num_h_steps
    )
    print("Num epoches", num_epochs)

    def _minimize_loop(training_state, state):
        synchro = pmap.is_replicated(
            (
                training_state.optimizer_state,
                training_state.params,
                training_state.normalizer_params,
            ),
            axis_name="i",
        )
        (training_state, state), losses = jax.lax.scan(
            run_epoch, (training_state, state), (), length=num_epochs // log_frequency
        )
        losses = jax.tree_map(jnp.mean, losses)
        return (training_state, state), losses, synchro

    minimize_loop = jax.pmap(_minimize_loop, axis_name="i")

    training_state = TrainingState(  # type: ignore
        optimizer_state=optimizer_state,
        params=init_params,
        key=jnp.stack(jax.random.split(key, local_devices_to_use)),
        normalizer_params=normalizer_params,
    )
    training_walltime = 0
    eval_walltime = 0
    sps = 0
    eval_sps = 0
    losses = {}
    state = first_state
    metrics = {}
    max_fitness = -jnp.inf

    for it in range(log_frequency + 1):
        logging.info("starting iteration %s %s", it, time.time() - xt)
        t = time.time()

        if process_id == 0:
            eval_state, rewards, dones, key_debug = run_eval(
                eval_first_state,
                key_debug,
                training_state.params["policy"],
                training_state.normalizer_params,
            )

            rewards_2 = rewards[:, 1:]
            rewards_2 = rewards_2.reshape(-1, num_eval_envs)
            dones = dones[:, 1:]
            dones = dones.reshape(-1, num_eval_envs)

            is_done = jnp.clip(jnp.cumsum(dones, axis=0), 0, 1)
            mask = jnp.roll(is_done, 1, axis=0)
            mask = mask.at[0, :].set(0)
            returns = jnp.sum(rewards_2 * (1.0 - mask), axis=0)

            max_return = jnp.max(returns)
            max_fitness = jnp.maximum(max_fitness, max_return)

            eval_metrics = eval_state.info["eval_metrics"]
            eval_metrics.completed_episodes.block_until_ready()
            eval_walltime += time.time() - t
            eval_sps = (
                episode_length * eval_first_state.reward.shape[0] / (time.time() - t)
            )
            avg_episode_length = (
                eval_metrics.completed_episodes_steps / eval_metrics.completed_episodes
            )
            metrics = dict(
                dict(
                    {
                        f"eval/episode_{name}": value / eval_metrics.completed_episodes
                        for name, value in eval_metrics.completed_episodes_metrics.items()
                    }
                ),
                **dict(
                    {
                        f"losses/{name}": jnp.mean(value)
                        for name, value in losses.items()
                    }
                ),
                **dict(
                    {
                        "eval/completed_episodes": eval_metrics.completed_episodes,
                        "eval/avg_episode_length": avg_episode_length,
                        "speed/sps": sps,
                        "speed/eval_sps": eval_sps,
                        "speed/training_walltime": training_walltime,
                        "speed/eval_walltime": eval_walltime,
                        "speed/timestamp": training_walltime,
                        "custom_eval/mean_return": jnp.mean(returns),
                        "custom_eval/max_return": jnp.max(returns),
                        "custom_eval/min_return": jnp.min(returns),
                        "custom_eval/std_return": jnp.std(returns),
                        "max_fitness": max_fitness,
                    }
                ),
            )
            logging.info(metrics)
            current_step = (
                int(training_state.normalizer_params[0][0])
                * action_repeat
                * num_h_steps
            )
            if progress_fn:
                progress_fn(current_step, metrics)

            if checkpoint_dir:
                normalizer_params = jax.tree_map(
                    lambda x: x[0], training_state.normalizer_params
                )
                h_policy_params = jax.tree_map(
                    lambda x: x[0], training_state.params["policy"]
                )
                params = normalizer_params, h_policy_params
                path = os.path.join(checkpoint_dir, f"ppo_{current_step}.pkl")
                model.save_params(path, params)

        if it == log_frequency:
            break

        t = time.time()
        previous_step = training_state.normalizer_params[0][0]
        # optimization
        (training_state, state), losses, synchro = minimize_loop(training_state, state)
        assert synchro[0], "break sync"
        jax.tree_map(lambda x: x.block_until_ready(), losses)
        sps = (
            (
                (training_state.normalizer_params[0][0] - previous_step)
                / (time.time() - t)
            )
            * action_repeat
            * num_h_steps
        )
        training_walltime += time.time() - t

    # To undo the pmap.
    normalizer_params = jax.tree_map(lambda x: x[0], training_state.normalizer_params)
    h_policy_params = jax.tree_map(lambda x: x[0], training_state.params["policy"])

    logging.info("total steps: %s", normalizer_params[0] * action_repeat * num_h_steps)

    inference = make_inference_fn(
        core_env.observation_size, core_env.action_size, normalize_observations
    )
    params = normalizer_params, h_policy_params

    pmap.synchronize_hosts()
    return (inference, params, metrics)


def make_inference_fn(observation_size, action_size, normalize_observations):
    """Creates params and inference function for the PPO agent."""
    _, obs_normalizer_apply_fn = normalization.make_data_and_apply_fn(
        observation_size, normalize_observations
    )
    h_parametric_action_distribution = Categorical(event_size=1)

    h_policy_model = networks.make_model(
        [
            32,
            32,
            32,
            32,
            action_size,
        ],
        observation_size,
        activation=jax.nn.relu,
    )

    def inference_fn(params, obs, key):
        normalizer_params, h_policy_params = params
        obs = obs_normalizer_apply_fn(normalizer_params, obs)
        action = h_parametric_action_distribution.sample(
            h_policy_model.apply(h_policy_params, obs), key
        )

        return action

    return inference_fn
