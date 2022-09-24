import matplotlib.pyplot as plt
import numpy as np

from .polar import Polar, open_polar

from .utils import radius, required_bank_for_radius
from .simulation import Simulation


def plot_xc_speed_thermal_mass(polar: Polar):
    netto_straight = np.array([2.0, -1.2])
    for netto_thermal in np.linspace(2.0, 5.0, 5):
        simulation = Simulation(netto_thermal, netto_straight)
        x = polar.mass + np.linspace(0, 150, 200)
        y = np.array([simulation.run(polar.for_mass(i)) for i in x])
        plt.plot(x, y * 3.6, label=f"Netto thermal strength: {netto_thermal:.1f} m/s")


def plot_xc_speed_mac_cready(polar: Polar):
    netto_thermal = 2.6
    netto_straight = np.array([2.0, -2.0])
    simulation = Simulation(netto_thermal, netto_straight)
    x = np.linspace(1.0, 3.0)
    y = np.array([simulation.run(polar.for_mass(polar.mass), i) for i in x])
    plt.plot(x, y * 3.6)


def plot_xc_speed_thermal(polar: Polar):
    netto_straight = np.array([0.0])
    x = np.linspace(1.0, 6.0)
    y = np.array(
        [Simulation(i, netto_straight).run(polar.for_mass(polar.mass)) for i in x]
    )
    plt.plot(x, y * 3.6, label=f"Glider: {polar.name}")


def plot_xc_speed_mass(polar: Polar):
    netto_thermal = 2.6
    netto_straight = np.array([2.0, 0.0, -2.0])
    simulation = Simulation(netto_thermal, netto_straight)
    x = polar.mass + np.linspace(0, 150, 200)
    y = np.array([simulation.run(polar.for_mass(i)) for i in x])
    plt.plot(x, y * 3.6)


def plot_max_bank_for_tas(polar: Polar):
    for alt in np.linspace(0, 5000, 5):
        x = np.linspace(20, 50)
        y = np.array([polar.max_bank_for_tas(alt, i) for i in x])
        plt.plot(x * 3.6, y, label=f"Alt: {alt:.0f} m")


def plot_sink_for_radius_speed(polar: Polar):
    for radius in np.linspace(80, 200, 6):
        x = np.array(
            [
                i
                for i in np.linspace(polar.min_speed, 40)
                if polar.min_speed_alt_bank(0, required_bank_for_radius(radius, i)) < i
            ]
        )
        y = np.array(
            [polar.evaluate(i, 0, required_bank_for_radius(radius, i)) for i in x]
        )
        plt.plot(x * 3.6, y, label=f"Radius: {radius} m")


def plot_speed_for_radius(polar: Polar):
    for radius in np.linspace(80, 200, 6):
        x = np.array(
            [
                i
                for i in np.linspace(polar.min_speed, 40)
                if polar.min_speed_alt_bank(0, required_bank_for_radius(radius, i)) < i
            ]
        )
        y = np.array([required_bank_for_radius(radius, i) for i in x])
        plt.plot(x * 3.6, y, label=f"Radius: {radius} m")


def plot_radius(polar: Polar):
    for phi in np.linspace(20, 50, 4):
        min_speed = polar.min_speed_alt_bank(0, phi)
        x = np.linspace(min_speed, 40)
        y = np.array([radius(i, phi) for i in x])
        plt.plot(x * 3.6, y, label=f"Bank: {phi}°")


def plot_min_speed_weight_bank(polar: Polar):
    # Min speed to fly at different weights and banks
    for mass_diff in np.linspace(0, 300, 5):
        new_polar = polar.for_mass(polar.mass + mass_diff)
        x = np.linspace(0, 60)
        y = np.array([new_polar.min_speed_alt_bank(0, i) * 3.6 for i in x])
        plt.plot(x, y, label=f"Mass: {new_polar.mass}")


def plot_min_speed_weight_altitude(polar: Polar):
    # Min speed to fly at different weights and altitude
    for mass_diff in np.linspace(0, 300, 5):
        new_polar = polar.for_mass(polar.mass + mass_diff)
        x = np.linspace(0, 6000)
        y = np.array([new_polar.min_speed_alt_bank(i, 0) * 3.6 for i in x])
        plt.plot(x, y, label=f"Mass: {new_polar.mass}")


def plot_at_altitude(polar: Polar):
    speed = polar.min_sink_speed  # m/s
    x = np.linspace(0, 10000)
    y = np.array([polar.evaluate(speed, alt=i, phi=0) for i in x])
    plt.plot(x, y, label=polar.name)


def plot_at_bank(polar: Polar):
    speed = polar.min_sink_speed  # m/s
    x = np.linspace(0, 60)
    y = np.array([polar.evaluate(speed, alt=0, phi=i) for i in x])
    plt.plot(x, y, label=polar.name)


def plot_speed_to_fly_weight(polar: Polar):
    # Speed to fly at different weights
    for mass_diff in np.linspace(0, 300, 5):
        new_polar = polar.for_mass(polar.mass + mass_diff)
        x = np.linspace(0, 5)
        y = np.array([new_polar.speed_to_fly(i) * 3.6 for i in x])
        plt.plot(x, y, label=polar.name)


def plot_speed_best_ld_speed_headwind(polar: Polar):
    # Best L/D at different weights and headwinds
    for mass_diff in np.linspace(0, 300, 5):
        new_polar = polar.for_mass(polar.mass + mass_diff)
        x = np.linspace(0, 20)
        y = np.array([new_polar.get_root(i, 0) * 3.6 for i in x])
        plt.plot(x, y, label=f"Mass: {new_polar.mass}")


def plot_speed_best_ld_headwind(polar: Polar):
    # Speed to fly at different weights and headwinds
    for mass_diff in np.linspace(0, 300, 5):
        new_polar = polar.for_mass(polar.mass + mass_diff)
        x = np.linspace(0, 20)
        y = np.array([new_polar.best_ld_headwind(i) for i in x])
        plt.plot(x, y, label=f"Mass: {new_polar.mass:.0f}")


def main():
    polars = (
        "LS4.POL",
        "LS3.POL",
        "LS3A.POL",
        "LS8.POL",
        "LS8_neo_2016.POL",
        "LS7.POL",
        # "LS6_1990.POL",
    )
    # polars = ("LS8.POL", "LS8_neo_2016.POhat L")
    polars = (
        # "LS4.POL",
        "LS8_neo_2016.POL",
        # "VENTUS2ct_18_new.POL",
        # "ASG32.POL",
        # "LS6C_18.POL",
        # "LS8_18.POL",
        # "JS-MD-3_18m.POL",
        # "ArcusT_2011.POL",
    )
    for polar_file in polars:
        polar_info, data = open_polar(polar_file)
        plt.plot([x[0] for x in data], [x[1] for x in data])
        polar = Polar.from_filename(polar_file)
        # polar.plt()
        # polar = polar.for_mass(800)
        # # plot_xc_speed_thermal_mass(polar)
        # plot_xc_speed_thermal(polar)

    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
