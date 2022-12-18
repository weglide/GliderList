import numpy as np

G = 9.81


def radius(tas: float, bank: float):
    """Radius that is achieved when flying with certain speed and bank"""
    return np.power(tas, 2) / (G * np.tan(np.radians(bank)))


def required_bank_for_radius(radius: float, speed: float):
    """Bank angle that must be flown to achieve a radius with a certain speed"""
    return np.degrees(np.arctan(np.power(speed, 2) / (G * radius)))
