"""Motor layouts. Order and spin directions follow ArduPilot's QUAD X and HEXA X numbering."""
from dataclasses import dataclass

import numpy as np

COUNTER_CLOCKWISE = +1
CLOCKWISE = -1


@dataclass(frozen=True)
class MotorLayout:
    name: str
    angles_from_nose_deg: tuple
    spin_directions: tuple

    @property
    def motor_count(self):
        return len(self.angles_from_nose_deg)

    def arm_offsets(self, arm_length_m):
        """Per-motor (forward, left) offsets from the centre of mass, metres."""
        angles = np.deg2rad(np.array(self.angles_from_nose_deg, dtype=np.float64))
        return arm_length_m * np.cos(angles), -arm_length_m * np.sin(angles)

    def spin_vector(self):
        return np.array(self.spin_directions, dtype=np.float64)


QUAD_X = MotorLayout("quad_x", (45, -135, -45, 135),
                     (COUNTER_CLOCKWISE, COUNTER_CLOCKWISE, CLOCKWISE, CLOCKWISE))
HEX_X = MotorLayout("hex_x", (90, -90, -30, 150, 30, -150),
                    (CLOCKWISE, COUNTER_CLOCKWISE, CLOCKWISE, COUNTER_CLOCKWISE, COUNTER_CLOCKWISE, CLOCKWISE))

LAYOUTS = {layout.name: layout for layout in (QUAD_X, HEX_X)}
LAYOUT_BY_MOTOR_COUNT = {layout.motor_count: layout for layout in LAYOUTS.values()}


def layout_named(name):
    if name not in LAYOUTS:
        raise ValueError(f"unknown frame {name!r}; choose from {sorted(LAYOUTS)}")
    return LAYOUTS[name]
