from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import NamedTuple

import numpy as np


class PolarPoint(NamedTuple):
    speed_ms: float  # m/s
    v_speed: float  # m/s


@dataclass
class Polar:
    coeffs: np.ndarray
    mass: float
    min_speed_ms: float
    name: str
    mtow: float | None = None
    wing_area: float | None = None

    @property
    def exponents(self) -> np.ndarray:
        return np.arange(len(self.coeffs) - 1, -1, -1)

    def __call__(self, x: float) -> float:
        return np.sum(self.coeffs * (x**self.exponents))

    @property
    def speeds(self) -> np.ndarray:
        return np.arange(self.min_speed_ms, 55, 0.1)  # type: ignore

    @property
    def v_speeds(self) -> np.ndarray:
        return np.array([self(s) for s in self.speeds])  # type: ignore

    def get_root(self, x_1: float, y_1: float) -> float:
        """Return the x-coordinate of the point on the polar whose tangent passes through (x_1, y_1).

        Solve the polynomial equation that a point on the polar needs to fulfill if his
        tangent passes through (x_1, y_1).

        Let (:math:`m = 2ax + b`) be the slope of the tangent.
        Let (:math:`(x_1, y_1)`) be the point we want to hit.
        We want to find (:math:`(x, y)`) to fulfill the following equation:

        .. math::
            y_1 + m * (x - x_1) = ax^2 + bx + c

        Args:
            x_1 (float): x-coordinate of the point that the tangent needs to pass through
            y_1 (float): y-coordinate of the point that the tangent needs to pass through

        Returns:
            float: x-coordinate of the point on the polar that the tangent passes through
        """
        m = (self.v_speeds - y_1) / (self.speeds - x_1)
        ix = np.argmax(m)
        return self.speeds[ix]

    @cached_property
    def best_ld(self) -> PolarPoint:
        speed_to_fly = self.get_root(0, 0)
        return PolarPoint(speed_to_fly, self(speed_to_fly))

    @cached_property
    def best_ld_value(self) -> float:
        return -self.best_ld.speed_ms / self.best_ld.v_speed

    def speed_to_fly(
        self, mac_cready: float, netto: float, headwind: float
    ) -> PolarPoint:
        speed = max(self.min_speed_ms, self.get_root(headwind, mac_cready - netto))
        return PolarPoint(speed, self(speed))
