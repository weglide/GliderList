from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, partial
from typing import Callable, NamedTuple

import matplotlib.pyplot as plt
import numpy as np

from .utils import radius, required_bank_for_radius


def density_factor(alt: float) -> float:
    ISA_LAPSE_RATE: float = 0.0065  # Lapse Rate of Standard Atmosphere
    ISA_TEMPERATURE: float = 288.15  # Temperature of Standard Atmosphere

    T: float = ISA_TEMPERATURE
    L: float = ISA_LAPSE_RATE

    K: float = 0.190266  # defined in comments

    return np.power(T / (T + L * alt), 1 / K)


GRAVITY = 9.81

POLAR_FOLDER = "./polars"

ENCODING = (0.48412173, 0.93094931, 0.12066504, 0.25892759, 0.52097507)

to_km_h = lambda speed: speed * 3.6
to_m_s = lambda speed: speed / 3.6


def radius(tas: float, bank: float):
    """Radius that is achieved when flying with certain speed and bank"""
    return np.power(tas, 2) / (GRAVITY * np.tan(np.radians(bank)))


def required_bank_for_radius(radius: float, speed: float):
    """Bank angle that must be flown to achieve a radius with a certain speed"""
    return np.degrees(np.arctan(np.power(speed, 2) / (GRAVITY * radius)))


class PolarPoint(NamedTuple):
    speed_ms: float  # m/s
    v_speed: float  # m/s

    @property
    def speed(self):
        return to_km_h(self.speed_ms)

    @property
    def ld(self):
        return -self.speed_ms / self.v_speed


class ThermalConfig(NamedTuple):
    speed_ms: float
    bank: float
    v_speed: float

    @property
    def speed(self):
        return to_km_h(self.speed_ms)

    def __str__(self):
        return (
            f"Config: {self.bank:.0f}°, {self.speed:.1f} km/h, {self.v_speed:.2f} m/s"
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
    index: int | None = None
    glider_class: str | None = None


@dataclass
class Polar:
    coeffs: np.ndarray
    mass: float
    min_speed_ms: float
    name: str
    mtow: float | None = None

    @property
    def save_min_speed_ms(self):
        return self.min_speed_ms + 1

    @property
    def exponents(self) -> np.ndarray:
        return np.arange(len(self.coeffs) - 1, -1, -1)

    def into_polar_ref(self, tas: float, phi: float, alt: float) -> float:
        """Move tas into the reference system of the polar(0m alt, 0° bank)

        Da im Kreisflug die Flächenbelastung hochgeht, muss die Geradeausfluggeschwindigkeit
        kleiner sein, als die IAS im Kreisflug

        Args:
            tas (float): True Airspeed (m/s)
            phi (float): Bank of the glider (degrees)
            alt (float): Altitude the glider is flying at (m)

        Returns:
            float: The equivalent airspeed at 0 m and 0° bank
        """
        ias = tas * np.sqrt(density_factor(alt))
        ias_polar = ias * np.sqrt(np.cos(np.radians(phi)))
        return ias_polar

    def from_polar_ref(self, ias_polar: float, phi: float, alt: float) -> float:
        """Reverse function of into_polar_ref

        Args:
            tas (float): True Airspeed (m/s)
            phi (float): Bank of the glider (degrees)
            alt (float): Altitude the glider is flying at (m)

        Returns:
            float: The equivalent airspeed at 0 m and 0° bank
        """
        ias = ias_polar / np.sqrt(np.cos(np.radians(phi)))
        tas = ias / np.sqrt(density_factor(alt))
        return tas

    def transform_v_speed(
        self,
        v_speed_level_0m: float,
        phi: float,
        alt: float,
    ) -> float:
        # W(phi=0,alt=0m) -> w(phi=X,alt=0m)
        # die v_speedgeschwindigkeit im Kreis muss höher sein als im Geradeausflug
        v_speed_bank_0m = v_speed_level_0m * np.power(np.cos(np.radians(phi)), -3 / 2)
        # W(phi=X,alt=0m) à w(phi=X,alt=Xm)
        # Die v_speedgeschwindigkeit im Kreis, in einer Höhe größer 0 muss höher sein.
        v_speed_bank_xm = v_speed_bank_0m / np.sqrt(density_factor(alt))

        return v_speed_bank_xm

    def evaluate(self, tas: float, phi: float, alt: int):
        ias_polar = self.into_polar_ref(tas, phi, alt)
        v_speed_polar = self(ias_polar)
        v_speed_bank_xm = self.transform_v_speed(v_speed_polar, phi, alt)
        return v_speed_bank_xm

    def netto(self, brutto: float, tas: float, phi: float, alt: int):
        return brutto - self.evaluate(tas, phi, alt)

    def __str__(self):
        return f"Mass: {self.mass} kg, L/D: {self.best_ld:.0f} @ {self.best_ld.speed:.0f} km/h, min sink {self.min_sink:.2f} m/s @ {self.min_sink.speed:.0f} km/h"

    def with_mass(self, new_mass: float) -> Polar:
        """Calculate new coefficients for different mass

        The increase in v_speed speed and glide speed is proportional to the square root of the increase in mass.
        The coefficients transform like:

        Args:
            new_mass (float): Mass of the new polar

        Returns:
            Polar: Newly generated polar
        """
        load_factor = np.sqrt(new_mass / self.mass)
        transformed = self.coeffs * load_factor ** (1 - self.exponents)
        min_speed_ms = load_factor * self.min_speed_ms
        return Polar(transformed, new_mass, min_speed_ms, self.name)

    def __call__(self, x: float) -> float:
        return np.sum(self.coeffs * (x**self.exponents))

    @property
    def speeds(self):
        return np.arange(self.min_speed_ms, 80, 0.1)

    @property
    def v_speeds(self):
        return np.array([self(s) for s in self.speeds])

    def get_root(self, x_1: float, y_1: float) -> float:
        """Return the point on the polar whose tangent passes through point.

        Solve the polynomial equation that a point on the polar needs to fullfill if his tangent
        passes throught point. This equation is obtained by starting on .

        Let (:math:`m = 2ax + b`) be the slope of the tangent. Let (:math:`(x_1, y_1)`) be the point we
        want to hit. Than we want to find (:math:`(x, y)`) to fullfill the following equation:

        .. math::
            y_1 + m * (x - x_1) = ax^2 + bx + c

        Args:
            point (Point): Point that the tangent needs to pass

        Returns:
            Point: Tangent point to the polar
        """
        m = (self.v_speeds - y_1) / (self.speeds - x_1)
        # ix = np.argmax(np.where(m < 0, m, -np.inf))
        ix = np.argmax(m)
        return self.speeds[ix]

    @cached_property
    def min_sink(self) -> PolarPoint:
        ix = np.argmax(self.v_speeds)
        return PolarPoint(self.speeds[ix], self.v_speeds[ix])

    @cached_property
    def best_ld(self) -> PolarPoint:
        speed_to_fly = self.get_root(0, 0)
        return PolarPoint(speed_to_fly, self(speed_to_fly))

    def speed_to_fly(
        self, mac_cready: float, netto: float, headwind: float
    ) -> PolarPoint:
        speed = max(self.min_speed_ms, self.get_root(headwind, mac_cready - netto))
        return PolarPoint(speed, self(speed))

    def thermal_config_for_radius(
        self, radius: float, alt: int
    ) -> ThermalConfig | None:
        config = None
        min_thermal_speed = self.min_speed_bank_alt(0, alt)
        thermal_speeds = np.linspace(min_thermal_speed, min_thermal_speed + 20, 300)
        for tas in thermal_speeds:
            rb = required_bank_for_radius(radius, tas)
            if rb > self.max_bank_for_tas(tas, alt):
                continue
            v_speed = self.evaluate(tas, rb, alt)
            if config is None or np.abs(v_speed) < np.abs(config.v_speed):
                config = ThermalConfig(tas, rb, v_speed)
        return config

    def min_speed_bank_alt(self, phi: float, alt: float) -> float:
        """Minium speed that can safely be flown at bank `phi` and altitude `alt`"""
        return self.from_polar_ref(self.save_min_speed_ms, phi, alt)

    def max_bank_for_tas(self, tas: float, alt: float) -> float:
        """Maxmium bank that can safely be flown at speed `tas` and altitude `alt`"""
        return np.degrees(
            np.arccos(
                np.power(
                    self.save_min_speed_ms / (tas * np.sqrt(density_factor(alt))), 2
                )
            )
        )

    def min_radius_for_speed(self, tas: float, alt: float) -> float:
        """Minimum radius that can safely be flown at speed `tas` and altitude `alt`"""
        max_bank = self.max_bank_for_tas(tas, alt)
        return radius(tas, max_bank)

    @staticmethod
    def ls_4() -> Polar:
        return Polar(
            coeffs=np.array([-0.0000115, -0.0017534, 0.1040406, -1.9786322]),
            mass=338.0,
            min_speed_ms=to_m_s(70.0),
            name="LS 4",
        )

    @classmethod
    def from_filename(cls, filename: str) -> Polar:
        info, data = open_polar(filename)
        return Polar.from_data_points(data, info.mass, to_m_s(info.min_speed), filename)

    @classmethod
    def from_data_points(
        cls, data: PolarData, mass: float, min_speed_ms: float, filename: str
    ) -> Polar:
        """Second degree polynomial regression to polar data."""
        x = to_m_s(np.array([p[0] for p in data]))
        y = np.array([p[1] for p in data])
        res = np.polyfit(x, y, 4)
        return Polar(
            coeffs=res,
            mass=mass,
            min_speed_ms=min_speed_ms,
            name=filename.split(".")[0],
        )

    def plt(self):
        x = np.arange(self.min_speed, 50)
        y = np.array([self(i) for i in x])
        plt.plot(x * 3.6, y)

    @property
    def encoded_coeffs(self) -> list[float]:
        running = 1
        data = []
        for e, coeff in zip(ENCODING, self.coeffs):
            running *= coeff
            data.append(running * e)
        return data

    def decode(self, encoded_coeffs: list[float]) -> list[float]:
        running = 1
        data = []
        for e, coeff in zip(ENCODING, encoded_coeffs):
            data.append(coeff / (running * e))
            running = coeff / e
        return data


Point = tuple[float, float]
PolarData = list[Point]


def polar_point(line: str) -> Point | None:
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


def open_polar(name: str) -> tuple[PolarInfo, PolarData]:
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

    if data_start is None:
        raise ValueError("No data found")

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
    polar_info = PolarInfo(**info)  # type: ignore
    return polar_info, data


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
        polar = polar.with_mass(525)


if __name__ == "__main__":
    main()
