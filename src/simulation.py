from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .polar import Polar


@dataclass
class Simulation:
    # thermals are available at all times
    netto_thermal: float
    # define of equally spaced conditions, step size does not matter
    netto_straight: np.ndarray
    thermal_radius: float = 130.0  # European average thermal, meter

    def run(self, polar: Polar, mac_cready: Optional[float] = None) -> Optional[float]:
        # use thermal_strength to gain needed altitude
        best_config = polar.thermal_config_for_radius(self.thermal_radius, 0)
        if best_config is None:
            return None
        if mac_cready is None:
            mac_cready = self.netto_thermal + best_config.sink
        speeds = [
            polar.speed_to_fly(mac_cready, netto) for netto in self.netto_straight
        ]
        # for speed, netto in zip(speeds, self.netto_straight):
        #     print(f"Flying {speed*3.6:.2f} km/h in {netto + polar(speed):.2f} m/s")

        time = [1 / s for s in speeds]
        alt_diff = sum(
            [(polar(s) + n) * t for s, n, t in zip(speeds, self.netto_straight, time)]
        )
        distance = len(self.netto_straight)

        if alt_diff > 0:
            raise Exception("Altitude diff must be negative")

        time_thermalling = -alt_diff / (self.netto_thermal + best_config.sink)
        xc_speed = distance / (sum(time) + time_thermalling)
        # print(f"XC Speed: {xc_speed * 3.6:.1f} km/h")
        # print(
        #     f"Time thermalling: {time_thermalling / (sum(time) + time_thermalling):.2f} %"
        # )
        return xc_speed

    def find_best_mac_cready(self, polar: float) -> float:
        mac_cready = 0.5
        step_size = 1.0
        best_speed = 0.0
        prev_speed = 0.0
        for _ in range(20):
            speed = self.run(polar, mac_cready=mac_cready)
            if speed > best_speed:
                best_speed = speed
            elif speed < prev_speed:
                step_size = -step_size / 2
            prev_speed = speed
            mac_cready += step_size

        return best_speed

    def find_best_mass(self, polar: Polar) -> Tuple[float, float, float]:
        step_size = 100.0
        best_speed = 0.0
        prev_speed = 0.0
        best_mass = polar.mass
        for _ in range(40):
            speed = self.find_best_mac_cready(polar)
            print(f"Speed {speed} with mass: {polar.mass}")
            if speed > best_speed:
                best_speed = speed
            elif speed < prev_speed:
                step_size = -step_size / 2
            prev_speed = speed
            polar = polar.for_mass(polar.mass + step_size)

        return best_speed, polar.mass


if __name__ == "__main__":
    polar = Polar.from_filename("LS4.POL")
    polar.plt()
    netto_thermal = 2.6
    mac_cready = netto_thermal - 0.6
    netto_straight = np.array([2.0, 0.0, -2.0])
    # netto_straight = np.array([0.0])
    simulation = Simulation(netto_thermal, netto_straight)
    for _ in range(8):
        polar = polar.for_mass(polar.mass + 30)
        speed = simulation.run(polar, mac_cready)
        if speed is None:
            print(f"Mass: {polar.mass} kg is too heavy")
            continue
        print(
            f"Speed: {speed * 3.6:.3f} km/h (MacCready: {mac_cready:.2f}, mass: {polar.mass} kg)\n"
        )
