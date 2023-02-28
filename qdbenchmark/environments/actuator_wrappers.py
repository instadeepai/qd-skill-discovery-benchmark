import copy

from brax.envs import Env, Wrapper
from qdax.environments.locomotion_wrappers import QDSystem


class ActuatorStrengthWrapper(Wrapper):
    def __init__(
        self, env: Env, actuator_name: str, strength_multiplier: float
    ) -> None:
        super().__init__(env)
        self._actuator_name = actuator_name
        self._strength_multiplier = strength_multiplier

        config = copy.copy(self.env.sys.config)

        actuators = config.actuators
        for actuator in actuators:
            if actuator.name == actuator_name:
                actuator.strength *= strength_multiplier

        self._config = config
        self.unwrapped.sys = QDSystem(self._config)
