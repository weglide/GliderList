from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Callable, List, NamedTuple, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from .utils import radius, required_bank_for_radius

POLAR_FOLDER = "./polars"


class ThermalConfig(NamedTuple):
    speed: float
    bank: float
    sink: float

    def __str__(self):
        return (
            f"Config: {self.bank:.0f}°, {self.speed*3.6:.1f} km/h, {self.sink:.2f} m/s"
        )


class PolarInfo(NamedTuple):
    name: str
    registration: str
    date: str  # do not parse this as format varies: `24/25.08.79` or `1979` or `16.8.81`
    wingspan: float
    wingarea: float
    mass: float
    wing_loading: float
    min_speed: float
    index: Optional[int] = None
    glider_class: Optional[str] = None


def density_factor(altitude: float) -> float:
    ISA_LAPSE_RATE: float = 0.0065  # Lapse Rate of Standard Atmosphere
    ISA_TEMPERATURE: float = 288.15  # Temperature of Standard Atmosphere

    T: float = ISA_TEMPERATURE
    L: float = ISA_LAPSE_RATE

    K: float = 0.190266  # defined in comments

    return np.power(T / (T + L * altitude), 1 / K)


@dataclass
class Polar:
    a: float
    b: float
    c: float
    mass: float
    # mtow: Optional[float] # TODO add
    min_speed: float  # = 22  # m/s = 79 km/h
    name: str

    @property
    def save_min_speed(self):
        return self.min_speed + 2  # 7.2 km/h above min speed

    def into_polar_ref(self, tas: float, phi: float, altitude: float) -> float:
        """Move tas into the reference system of the polar(0m altitude, 0° bank)

        Da im Kreisflug die Flächenbelastung hochgeht, muss die Geradeausfluggeschwindigkeit
        kleiner sein, als die IAS im Kreisflug

        Args:
            tas (float): True Airspeed (m/s)
            phi (float): Bank of the glider (degrees)
            altitude (float): Altitude the glider is flying at (m)

        Returns:
            float: The equivalent airspeed at 0 m and 0° bank
        """
        ias = tas * np.sqrt(density_factor(altitude))
        ias_polar = ias * np.sqrt(np.cos(np.radians(phi)))
        return ias_polar

    def from_polar_ref(self, ias_polar: float, phi: float, altitude: float) -> float:
        """Reverse function of into_polar_ref

        Args:
            tas (float): True Airspeed (m/s)
            phi (float): Bank of the glider (degrees)
            altitude (float): Altitude the glider is flying at (m)

        Returns:
            float: The equivalent airspeed at 0 m and 0° bank
        """
        ias = ias_polar / np.sqrt(np.cos(np.radians(phi)))
        tas = ias / np.sqrt(density_factor(altitude))
        return tas

    def transform_sink(
        self,
        sink_level_0m: float,
        phi: float,
        altitude: float,
    ) -> float:
        # W(phi=0,alt=0m) -> w(phi=X,alt=0m)
        # die Sinkgeschwindigkeit im Kreis muss höher sein als im Geradeausflug
        sink_bank_0m = sink_level_0m * np.power(np.cos(np.radians(phi)), -3 / 2)
        # W(phi=X,alt=0m) à w(phi=X,alt=Xm)
        # Die Sinkgeschwindigkeit im Kreis, in einer Höhe größer 0 muss höher sein.
        sink_bank_xm = sink_bank_0m / np.sqrt(density_factor(altitude))

        return sink_bank_xm

    def evaluate(self, tas: float, phi: float, altitude: int):
        ias_polar = self.into_polar_ref(tas, phi, altitude)
        sink_polar = self(ias_polar)
        sink_bank_xm = self.transform_sink(sink_polar, phi, altitude)
        return sink_bank_xm

    def __str__(self):
        return f"Mass: {self.mass} kg, L/D: {self.best_ld:.0f} @ {self.best_ld_speed*3.6:.0f} km/h, min sink {self.min_sink:.2f} m/s @ {self.min_sink_speed * 3.6:.0f} km/h"

    def plt(self):
        x = np.arange(self.min_speed, 50)
        y = np.array([self(i) for i in x])
        plt.plot(x * 3.6, y)

    def for_mass(self, new_mass: float) -> Polar:
        """Calculate new coefficients for different mass

        The increase in sink speed and glide speed is proportional to the square root of the increase in mass.
        The coefficients transform like:

        .. math::
            \lambda y = \tilde{a}(\lambda x)^2 + \tilde{b}\lambda x + \tilde{c}
            \begin{align*}
                \tilde{a} &= \frac{a}{\lambda} \\
                \tilde{b} &= b \\
                \tilde{c} &= c\lambda
            \end{align*}

        Args:
            new_mass (float): Mass of the new polar

        Returns:
            Polar: Newly generated polar
        """
        factor = np.sqrt(new_mass / self.mass)
        a = self.a / factor
        b = self.b
        c = self.c * factor
        min_speed = factor * self.min_speed
        return Polar(a, b, c, new_mass, min_speed, self.name)

    def __call__(self, x: float) -> float:
        return self.a * x**2 + self.b * x + self.c

    def get_root(self, x_1, y_1) -> Point:
        """Return the point on the polar whose tangent passes through point.

        Solve the polynomial equation that a point on the polar needs to fullfill if his tangent
        passes throught point. This equation is obtained by starting on .

        Let (:math:`m = 2ax + b`) be the slope of the tangent. Let (:math:`(x_1, y_1)`) be the point we
        want to hit. Than we want to find (:math:`(x, y)`) to fullfill the following equation:

        .. math::
            y_1 + m * (x - x_1) = ax^2 + bx + c

        which simplifies to

        .. math::
            ax^2 - 2ax_1x + y_1 - c - bx_1


        Args:
            point (Point): Point that the tangent needs to pass

        Returns:
            Point: Tangent point to the polar
        """
        a = self.a
        b = -2 * self.a * x_1
        c = y_1 - self.c - self.b * x_1

        poly = np.polynomial.Polynomial((c, b, a))
        roots = poly.roots()
        return roots[1]

    @property
    def min_sink_speed(self) -> float:
        return max(self.min_speed, -self.b / (2 * self.a))

    @property
    def min_sink(self) -> float:
        return self(self.min_sink_speed)

    @property
    def best_ld_speed(self) -> float:
        return self.get_root(0, 0)

    @property
    def best_ld(self) -> float:
        return -self.best_ld_speed / self(self.best_ld_speed)

    def best_ld_headwind(self, headwind: float) -> float:
        speed_to_fly = self.get_root(headwind, 0)
        return -(speed_to_fly - headwind) / self(speed_to_fly)

    def to_netto(self, brutto: float) -> float:
        pass

    def speed_to_fly(self, mac_cready: float, netto: float) -> float:
        return max(self.min_speed, self.get_root(0, mac_cready - netto))

    def thermal_config_for_radius(
        self, radius: float, altitude: float
    ) -> Optional[ThermalConfig]:
        config = None
        for tas in np.linspace(
            self.min_speed_alt_bank(0, altitude), self.min_speed + 10, 300
        ):
            rb = required_bank_for_radius(radius, tas)
            # print(f"Speed {tas*3.6:.2f} km/h requires bank of {rb:.2f}°")
            if rb > self.max_bank_for_tas(tas, altitude):
                continue
            sink = self.evaluate(tas, rb, altitude)
            if config is None or np.abs(sink) < np.abs(config.sink):
                config = ThermalConfig(tas, rb, sink)
        return config

    def min_speed_alt_bank(self, phi: float, altitude: float) -> float:
        return self.from_polar_ref(self.save_min_speed, phi, altitude)

    def max_bank_for_tas(self, tas: float, altitude: float) -> float:
        return np.degrees(
            np.arccos(
                np.power(
                    self.save_min_speed / (tas * np.sqrt(density_factor(altitude))), 2
                )
            )
        )

    def min_radius_for_speed(self, tas: float, altitude: float) -> float:
        max_bank = self.max_bank_for_tas(tas, altitude)
        return radius(tas, max_bank)

    @classmethod
    def from_filename(cls, filename: str) -> Polar:
        info, data = open_polar(filename)
        return Polar.from_data_points(data, info.mass, info.min_speed / 3.6, filename)

    @classmethod
    def from_data_points(
        cls, data: PolarData, mass: float, min_speed: float, filename: str
    ) -> Polar:
        """Second degree polynomial regression to polar data."""
        # km/h -0.0002227, 0.040399, -2.48156
        # m/s -0.002886, 0.1454, -2.4815
        # from FB a 0.1037, b -8.8112 c 301.56
        # For an unloaded JS1 at 35.7 kg/sq.m the parameters are: a = +1.54, b = -2.81, and c = 1.85
        x = np.array([p[0] for p in data])
        x = x / 3.6
        y = np.array([p[1] for p in data])
        res = np.polyfit(x, y, 2)
        return Polar(
            a=res[0],
            b=res[1],
            c=res[2],
            mass=mass,
            min_speed=min_speed,
            name=filename.split(".")[0],
        )


Point = Tuple[float, float]
PolarData = List[Tuple[Point]]


def polar_point(line: str) -> Optional[Point]:
    entries = line.split()
    if not entries:
        return None
    try:
        x, y = float(entries[0]), float(entries[1])
    except (ValueError, IndexError):
        return None
    return x, y


def save_func(func: Callable, input: str):
    try:
        return func(input)
    except ValueError:
        return input


save_float = partial(save_func, float)


def open_polar(name: str) -> Tuple[PolarInfo, PolarData]:
    data = []
    with open(POLAR_FOLDER + "/" + name) as file:
        # print(f"Parsing {name} ...")
        lines = [l.rstrip() for l in file.readlines()]

    data_start = None
    for i, line in enumerate(lines):
        point = polar_point(line)
        if point is not None:
            if data_start is None:
                data_start = i
            data.append(point)

    fields = [
        "name",
        "registration",
        "date",
        "wingspan",
        "wingarea",
        "mass",
        "wing_loading",
        "min_speed",
        "index",
        "glider_class",
    ]
    info = {k: save_float(lines[i]) for i, k in enumerate(fields) if i < data_start}
    if info.get("min_speed") is None:
        info["min_speed"] = data[0][0]
    info = PolarInfo(**info)
    return info, data


def main():
    # polars = ("LS4.POL", "LS3.POL", "LS8.POL", "LS7.POL", "LS6_1990.POL")
    # polars = ("LS8.POL", "LS8_neo_2016.POL")
    polars = (
        "LS8_neo_2016.POL",
        # "VENTUS2ct_18_new.POL",
        # "ASG32.POL",
        # "LS6C_18.POL",
        # "JS-MD-3_18m.POL",
    )
    for polar_file in polars:
        polar = Polar.from_filename(polar_file)
        polar = polar.for_mass(525)
        # print(f"Min speed: {polar.min_speed_alt_bank(0, 40) * 3.6:.2f}")


if __name__ == "__main__":
    main()
