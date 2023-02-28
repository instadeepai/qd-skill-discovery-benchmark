import copy

import brax
import jax.numpy as jnp
from brax.envs import env


class DefaultPositionWrapper(env.Wrapper):
    def __init__(
        self, env: env.Env, body_name: str, body_position: jnp.ndarray
    ) -> None:
        super().__init__(env)

        self._body_name = body_name
        self._body_position = body_position
        config = copy.copy(self.env.sys.config)

        for default in config.defaults:
            for default_qp in default.qps:
                if default_qp.name == body_name:
                    default_qp.pos.x = body_position[0]
                    default_qp.pos.y = body_position[1]
                    default_qp.pos.z = body_position[2]

        self._config = config
        self.unwrapped.sys = brax.System(self._config)
