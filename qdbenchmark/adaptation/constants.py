from typing import Dict

import numpy as np

ACTUATOR_UPDATES = {
    "walker2d_uni": {
        "thigh_left_joint": np.array([0.25 * i for i in range(20)]),
        "leg_left_joint": np.array([0.25 * i for i in range(20)]),
        "foot_left_joint": np.array([0.25 * i for i in range(20)]),
    },
    "ant_uni": {
        "$ Torso_Aux 4": np.array([0.25 * i for i in range(20)]),
        "Aux 4_$ Body 13": np.array([0.25 * i for i in range(20)]),
    },
    "halfcheetah_uni": {
        "fthigh": np.array([0.25 * i for i in range(20)]),
        "fshin": np.array([0.25 * i for i in range(20)]),
        "ffoot": np.array([0.25 * i for i in range(20)]),
    },
}

GRAVITY_MULTIPLIERS = {
    "walker2d_uni": np.array([0.25 * (i + 1) for i in range(20)]),
    "ant_uni": np.array([0.25 * (i + 1) for i in range(20)]),
    "halfcheetah_uni": np.array([0.25 * (i + 1) for i in range(20)]),
}


TARGET_POSITIONS = np.array(
    [
        [11.739233, 35.159004, 0.5],
        [-2.3251667, 14.860174, 0.5],
        [32.481056, 32.365906, 0.5],
        [29.523472, 17.460098, 0.5],
        [12.460787, 7.0494328, 0.5],
        [17.36067, 4.448854, 0.5],
        [8.380336, 25.172598, 0.5],
        [24.024515, 34.360985, 0.5],
        [21.746223, 14.116749, 0.5],
        [19.50973, 14.577911, 0.5],
    ],
    dtype=np.float32,
)

ANTMAZE_TARGET_POSITIONS = {"Target": TARGET_POSITIONS}

ADAPTATION_CONSTANTS: Dict[str, dict] = {
    "gravity_multiplier": GRAVITY_MULTIPLIERS,
    "actuator_update": ACTUATOR_UPDATES,
    "default_target_position": ANTMAZE_TARGET_POSITIONS,
}
