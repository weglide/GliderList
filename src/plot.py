import matplotlib.pyplot as plt
import numpy as np

from .polar import Polar, open_polar

from .utils import radius, required_bank_for_radius
from .simulation import Simulation


def plot_optimal_climb(polar: Polar):
    polar = polar.with_mass(400)
    for radius in np.linspace(80, 220, 5):
        speeds = np.array([
            s
            for s in np.linspace(polar.min_speed_ms, polar.min_speed_ms + 20, 100)
            if polar.min_speed_bank_alt(required_bank_for_radius(radius, s), 0) < s
        ])
        sink = [
            polar.evaluate(tas, required_bank_for_radius(radius, tas), 0)
            for tas in speeds
        ]
        best = polar.thermal_config_for_radius(radius, 0)
        assert best is not None
        plt.plot(best.speed * 3.6, best.v_speed, "-ro")
        plt.plot(speeds * 3.6, sink, label=f"Radius: {radius:.1f} m")

    plt.xlabel("Speed [km/h]")
    plt.ylabel("Sink [m/s]")
    plt.title("Thermal Config")


def plot_xc_speed_thermal_mass(polar: Polar):
    netto_straight = np.array([2.0, -1.2])
    for netto_thermal in np.linspace(2.0, 6.0, 5):
        simulation = Simulation(netto_thermal, netto_straight)
        x = polar.mass + np.linspace(0, 150, 200)
        y = np.array([simulation.run(polar.with_mass(i)) for i in x])
        plt.plot(x, y * 3.6, label=f"Netto thermal strength: {netto_thermal:.1f} m/s")
    plt.xlabel("Mass [kg]")
    plt.ylabel("Achievable XC speed [km/h]")
    plt.title("Achievable XC speed for different mass")


def plot_xc_speed_mac_cready(polar: Polar):
    polar = polar.with_mass(525)
    for netto_thermal in np.linspace(2.0, 6.0, 5):
        netto_straight = np.array([-0.5])
        simulation = Simulation(netto_thermal, netto_straight)
        mac_cready = np.linspace(1.0, 4.0)
        y = np.array(
            [simulation.run(polar.with_mass(polar.mass), m) for m in mac_cready]
        )
        plt.plot(mac_cready, y * 3.6, label=f"Netto Thermal: {netto_thermal:.1f}")
    plt.xlabel("MacCready setting [m/s]")
    plt.ylabel("Achievable XC speed [km/h]")
    plt.title("Achievable XC speed for different MacCready settings")


def plot_xc_speed_thermal(polar: Polar):
    for mass_diff in np.linspace(0, 200, 5):
        new_polar = polar.with_mass(polar.mass + mass_diff)
        netto_straight = np.array([0.6])
        x = np.linspace(1.0, 6.0)
        y = np.array([Simulation(i, netto_straight).run(new_polar) or 0.0 for i in x])
        plt.plot(x, y * 3.6, label=f"Glider: {polar.name} at {new_polar.mass} kg")
    plt.xlabel("Average climb values [m/s]")
    plt.ylabel("Achievable XC speed [km/h]")
    plt.title("Achievable XC speed for different climb values")


def plot_xc_speed_mass(polar: Polar):
    netto_thermal = 2.6
    netto_straight = np.array([2.0, 0.0, -2.0])
    simulation = Simulation(netto_thermal, netto_straight)
    x = polar.mass + np.linspace(0, 150, 200)
    y = np.array([simulation.run(polar.with_mass(i)) for i in x])
    plt.plot(x, y * 3.6)
    plt.xlabel("Mass of glider [kg]")
    plt.ylabel("Achievable XC speed [km/h]")
    plt.title("Achievable XC speed for different mass")


def plot_max_bank_for_tas(polar: Polar):
    for alt in np.linspace(0, 5000, 5):
        tas = np.linspace(20, 50)
        y = np.array([polar.max_bank_for_tas(t, alt) for t in tas])
        plt.plot(tas * 3.6, y, label=f"Alt: {alt:.0f} m")
    plt.xlabel("True Airspeed [km/h]")
    plt.ylabel("Maximum Bank [°]")
    plt.title("Maximum bank of glider for different altitudes")


def plot_sink_for_radius_speed(polar: Polar):
    for radius in np.linspace(80, 200, 6):
        print(radius)
        x = np.array(
            [
                i
                for i in np.linspace(polar.min_speed_ms, polar.min_speed_ms + 20, 300)
                if polar.min_speed_bank_alt(required_bank_for_radius(radius, i), 0) < i
            ]
        )
        y = np.array(
            [polar.evaluate(i, required_bank_for_radius(radius, i), 0) for i in x]
        )
        plt.plot(x * 3.6, y, label=f"Radius: {radius} m")
    plt.xlabel("Speed [km/h]")
    plt.ylabel("Sink [m/s]")
    plt.title("Sink of glider at different radii and speed")


def plot_speed_for_radius(polar: Polar):
    for radius in np.linspace(80, 200, 6):
        x = np.array(
            [
                i
                for i in np.linspace(polar.min_speed_ms, 40)
                if polar.min_speed_bank_alt(required_bank_for_radius(radius, i), 0) < i
            ]
        )
        y = np.array([required_bank_for_radius(radius, i) for i in x])
        plt.plot(x * 3.6, y, label=f"Radius: {radius} m")
    plt.xlabel("Speed [km/h]")
    plt.ylabel("Required Bank [°]")
    plt.title("Required speed and bank for different radii")


def plot_radius(polar: Polar):
    for phi in np.linspace(20, 50, 4):
        min_speed_ms = polar.min_speed_bank_alt(phi, 0)
        x = np.linspace(min_speed_ms, 40)
        y = np.array([radius(i, phi) for i in x])
        plt.plot(x * 3.6, y, label=f"Bank: {phi}°")
    plt.xlabel("Speed [km/h]")
    plt.ylabel("Radius [m]")
    plt.title("Radius for different speed and bank")


def plot_min_speed_weight_bank(polar: Polar):
    for mass_diff in np.linspace(0, 300, 5):
        new_polar = polar.with_mass(polar.mass + mass_diff)
        x = np.linspace(0, 60)
        y = np.array([new_polar.min_speed_bank_alt(i, 0) * 3.6 for i in x])
        plt.plot(x, y, label=f"Mass: {new_polar.mass}")
    plt.xlabel("Bank [°]")
    plt.ylabel("Min Speed [km/h]")
    plt.title("Min speed to fly at different weights and banks")


def plot_min_speed_weight_altitude(polar: Polar):
    for mass_diff in np.linspace(0, 300, 5):
        new_polar = polar.with_mass(polar.mass + mass_diff)
        x = np.linspace(0, 6000)
        y = np.array([new_polar.min_speed_bank_alt(0, i) * 3.6 for i in x])
        plt.plot(x, y, label=f"Mass: {new_polar.mass}")
    plt.xlabel("Altitude [m]")
    plt.ylabel("Min Speed [km/h]")
    plt.title("Min speed to fly at different weights and altitude")


def plot_at_altitude(polar: Polar):
    speed = polar.min_sink.speed_ms  # m/s
    x = np.linspace(0, 10000)
    y = np.array([polar.evaluate(speed, alt=i, phi=0) for i in x])
    plt.plot(x, y, label=polar.name)
    plt.xlabel("Altitude [m]")
    plt.ylabel("Sink [m/s]")
    plt.title("Sink at different altitudes")


def plot_at_bank(polar: Polar):
    speed = polar.min_sink.speed_ms
    x = np.linspace(0, 60)
    y = np.array([polar.evaluate(speed, alt=0, phi=i) for i in x])
    plt.plot(x, y, label=polar.name)
    plt.xlabel("Bank [°]")
    plt.ylabel("Sink [m/s]")
    plt.title("Sink at different bank angles")


def plot_speed_to_fly_weight(polar: Polar):
    for mass_diff in np.linspace(0, 300, 5):
        new_polar = polar.with_mass(polar.mass + mass_diff)
        x = np.linspace(0, 5)
        y = np.array([new_polar.speed_to_fly(i, 0.0, 0.0).speed_ms * 3.6 for i in x])
        plt.plot(x, y, label=f"Mass: {new_polar.mass:.0f} kg")
    plt.xlabel("MacCready [m/s]")
    plt.ylabel("Speed to fly [km/h]")
    plt.title("Speed to fly at different weights")


def plot_speed_best_ld_speed_headwind(polar: Polar):
    for mass_diff in np.linspace(0, 300, 5):
        new_polar = polar.with_mass(polar.mass + mass_diff)
        x = np.linspace(-20, 20)
        y = np.array([new_polar.get_root(i, 0) * 3.6 for i in x])
        plt.plot(x * 3.6, y, label=f"Mass: {new_polar.mass:.0f} kg")
    plt.xlabel("Headwind [km/h]")
    plt.ylabel("Speed to fly [km/h]")
    plt.title("Best L/D at different weights and headwinds")


def plot_polar(polar: Polar):
    items = 350
    plt.plot(polar.speeds[:items] * 3.6, polar.v_speeds[:items], label="Fit")

    plt.xlabel("Speed [km/h]")
    plt.ylabel("Sink [m/s]")


def main():
    polar_file = "LS8_neo_2016.POL"

    # plot the actual data
    polar_info, data = open_polar(polar_file)
    plt.plot([d[0] for d in data], [d[1] for d in data], label="Measured")

    polar = Polar.from_filename(polar_file)
    plot_polar(polar)

    plt.legend()
    plt.show()


def compare_with_xcsoar():
    polar = Polar(
    #   np.array([-0.0000002, -0.0000033, 0.0000181, 0.0006924, -0.3305156]),  # nimbus
      np.array([-0.0000022, 0.0003027, -0.0178467, 0.4667099, -5.0310928]),  # ls7
      597,
      65.10 / 3.6,
      name="Nimbus 4",
      mtow=850,
    )

    plot_polar(polar)
    plt.legend()
    plt.show()


def xcsoar_polyfit(v: list[float], w: list[float]) -> tuple[float, float, float]:
    v1, v2, v3 = v
    w1, w2, w3 = w
    d = v1 * v1 * (v2 - v3) + v2 * v2 * (v3 - v1) + v3 * v3 * (v1 - v2)
    if d == 0:
        a = 0.0
    else:
        a = -((v2 - v3) * (w1 - w3) + (v3 - v1) * (w2 - w3)) / d

    d = v2 - v3
    if d == 0:
        b = 0.0
    else:
        b = -(w2 - w3 + a * (v2 * v2 - v3 * v3)) / d

    c = -(w3 + a * v3 * v3 + b * v3)
    return a, b, c



def coeff_to_lx(a, b, c):
    b = b * 100 / 3.6
    a = a * (100 / 3.6) ** 2
    return -a, -b, -c


def coeff_from_lx(a, b, c):
    b = b / (100 / 3.6)
    a = a / (100 / 3.6) ** 2
    return -a, -b, -c



if __name__ == "__main__":
    # main()
    # plot three points
    x = np.array([103.77,  155.65,  180.00]) / 3.6
    y = np.array([-0.73, -1.47, -2.66])


    # x = np.array([85.1, 127.98, 162.74]) / 3.6
    # y = np.array([-0.41, -0.75, -1.4])


    # it's the same
    res = np.polyfit(x, y, 2)
    z = np.arange(80, 180) / 3.6
    zz = res[0] * z**2 + res[1] * z + res[2]
    plt.plot(z * 3.6, zz, label="Simple Fit")

    res = coeff_from_lx(1.78, -3.03, 1.93)
    zz = res[0] * z**2 + res[1] * z + res[2]
    plt.plot(z * 3.6, zz, label="LX Polar")

    plt.plot(x * 3.6, y, label="XCSoar")
    plt.show()
    plt.legend()


    # compare_with_xcsoar()
