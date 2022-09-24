import numpy as np

G = 9.81


def radius(speed: float, bank: float):
    return np.power(speed, 2) / (G * np.tan(np.radians(bank)))


def required_bank_for_radius(radius: float, speed: float):
    return np.degrees(np.arctan(np.power(speed, 2) / (G * radius)))
