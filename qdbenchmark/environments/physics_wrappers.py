import copy

from brax.envs import Env, Wrapper
from qdax.environments.locomotion_wrappers import QDSystem


class GravityWrapper(Wrapper):
    def __init__(self, env: Env, gravity_multiplier: float) -> None:
        super().__init__(env)
        self._gravity_multiplier = gravity_multiplier
        config = copy.copy(self.env.sys.config)
        config.gravity.z *= gravity_multiplier
        self._config = config

        self.unwrapped.sys = QDSystem(self._config)


class FrictionWrapper(Wrapper):
    def __init__(self, env: Env, friction_multiplier: float) -> None:
        super().__init__(env)
        self._friction_multiplier = friction_multiplier
        config = copy.copy(self.env.sys.config)

        config.friction *= friction_multiplier
        self._config = config

        self.unwrapped.sys = QDSystem(self._config)
