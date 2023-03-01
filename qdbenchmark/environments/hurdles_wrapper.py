import jax.numpy as jnp
from google.protobuf import text_format  # type: ignore
from qdax.environments.locomotion_wrappers import COG_NAMES

import brax
from brax import jumpy as jp
from brax.envs import State, env

# config of the body of the trap following Brax's config style
HURDLES_CONFIG = """bodies {
    name: "Hurdles"
    colliders {
        position { x: 3.0 y: 0.0 z: 0.25 }
        rotation { x: 90 y: 0 }
        capsule {
            radius: 0.25
            length: 1
            end: 0
        }
    }
    colliders {
        position { x: 6.0 y: 0.0 z: 0.25 }
        rotation { x: 90 y: 0 }
        capsule {
            radius: 0.25
            length: 1
            end: 0
        }
    }
    colliders {
        position { x: 9.0 y: 0.0 z: 0.25 }
        rotation { x: 90 y: 0 }
        capsule {
            radius: 0.25
            length: 1
            end: 0
        }
    }
    colliders {
        position { x: 12.0 y: 0.0 z: 0.25 }
        rotation { x: 90 y: 0 }
        capsule {
            radius: 0.25
            length: 1
            end: 0
        }
    }
    colliders {
        position { x: 15.0 y: 0.0 z: 0.25 }
        rotation { x: 90 y: 0 }
        capsule {
            radius: 0.25
            length: 1
            end: 0
        }
    }
    colliders {
        position { x: 18.0 y: 0.0 z: 0.25 }
        rotation { x: 90 y: 0 }
        capsule {
            radius: 0.25
            length: 1
            end: 0
        }
    }
    colliders {
        position { x: 21.0 y: 0.0 z: 0.25 }
        rotation { x: 90 y: 0 }
        capsule {
            radius: 0.25
            length: 1
            end: 0
        }
    }
    colliders {
        position { x: 24.0 y: 0.0 z: 0.25 }
        rotation { x: 90 y: 0 }
        capsule {
            radius: 0.25
            length: 1
            end: 0
        }
    }
    colliders {
        position { x: 27.0 y: 0.0 z: 0.25 }
        rotation { x: 90 y: 0 }
        capsule {
            radius: 0.25
            length: 1
            end: 0
        }
    }
    colliders {
        position { x: 30.0 y: 0.0 z: 0.25 }
        rotation { x: 90 y: 0 }
        capsule {
            radius: 0.25
            length: 1
            end: 0
        }
    }
    inertia { x: 10000.0 y: 10000.0 z: 10000.0 }
    mass: 1
    frozen { all: true }
}
"""


# config describing collisions of the trap following Brax's config style
# specific to the ant env
HALFCHEETAH_HURDLES_COLLISIONS = """collide_include {
    first: "torso"
    second: "Hurdles"
}
collide_include {
    first: "bfoot"
    second: "Hurdles"
}
collide_include {
    first: "ffoot"
    second: "Hurdles"
}
collide_include {
    first: "bthigh"
    second: "Hurdles"
}
collide_include {
    first: "fthigh"
    second: "Hurdles"
}
collide_include {
    first: "bshin"
    second: "Hurdles"
}
collide_include {
    first: "fshin"
    second: "Hurdles"
}
collide_include {
    first: "Hurdles"
    second: "Ground"
}
"""

# storing the classic env configurations
# those are the configs from the official brax repo
ENV_SYSTEM_CONFIG = {
    "halfcheetah": brax.envs.halfcheetah._SYSTEM_CONFIG_SPRING,
}

# linking each env with its specific collision description
# could made more automatic in the future
ENV_HURDLES_COLLISION = {
    "halfcheetah": HALFCHEETAH_HURDLES_COLLISIONS,
}


class HurdlesWrapper(env.Wrapper):
    def __init__(self, env: env.Env, env_name: str) -> None:
        if (
            env_name not in ENV_SYSTEM_CONFIG.keys()
            or env_name not in COG_NAMES.keys()
            or env_name not in ENV_HURDLES_COLLISION.keys()
        ):
            raise NotImplementedError(f"This wrapper does not support {env_name} yet.")

        super().__init__(env)
        self._env_name = env_name
        # update the env config to add the trap
        self._config = (
            ENV_SYSTEM_CONFIG[env_name]
            + HURDLES_CONFIG
            + ENV_HURDLES_COLLISION[env_name]
        )
        # update the associated physical system
        config = text_format.Parse(self._config, brax.Config())
        if not hasattr(self.unwrapped, "sys"):
            raise AttributeError("Cannot link env to a physical system.")
        self.unwrapped.sys = brax.System(config)
        self._cog_idx = self.unwrapped.sys.body.index[COG_NAMES[env_name]]

        # we need to normalise x/y position to avoid values to explose
        self._substract = jnp.array([15, 0])  # come from env limits
        self._divide = jnp.array([15, 8])  # come from env limits

    @property
    def name(self) -> str:
        return self._env_name

    @property
    def observation_size(self) -> int:
        """The size of the observation vector returned in step and reset."""
        rng = jp.random_prngkey(0)
        reset_state = self.reset(rng)
        return int(reset_state.obs.shape[-1])

    def reset(self, rng: jp.ndarray) -> State:
        state = self.env.reset(rng)
        state.info["state_descriptor"] = jnp.nan
        # add xy position to the observation
        xy_pos = state.qp.pos[self._cog_idx][:2]
        # normalise
        xy_pos = (xy_pos - self._substract) / self._divide
        new_obs = jp.concatenate([xy_pos, state.obs])
        return state.replace(obs=new_obs)  # type: ignore

    def step(self, state: State, action: jp.ndarray) -> State:
        state = self.env.step(state, action)
        state.info["state_descriptor"] = jnp.nan
        # add xy position to the observation
        xy_pos = state.qp.pos[self._cog_idx][:2]
        # normalise
        xy_pos = (xy_pos - self._substract) / self._divide
        new_obs = jp.concatenate([xy_pos, state.obs])
        return state.replace(obs=new_obs)  # type: ignore
