import numpy as np
import csv
from src.polar import Polar


def add_bugs(coeffs: np.ndarray, bugs: float):
    return coeffs * (1 + bugs)


def get_polar(name: str) -> Polar | None:
    with open("data/gliderlist_merged.csv") as file:
        reader = csv.reader(file, delimiter=",")
        for row in reader:
            if row[2] == name:
                data = row[11].split(":")
                print(data)
                coeffs = np.array([float(data[i]) for i in range(0, len(data))])
                mass = float(row[8])
                min_speed_ms = float(row[10]) / 3.6
                mtow = float(row[9])
                return Polar(coeffs, mass, min_speed_ms, name, mtow)
    return None


def test_speed_to_fly_polar():
    polar = Polar.from_filename("LS3A78.POL")
    polar.with_mass(280.0)
    data = [(0.0, 100), (0.5, 116), (1.0, 129), (2.0, 153), (3.0, 174), (4.0, 193)]
    for mac_cready, speed_to_fly in data:
        assert (
            round(polar.speed_to_fly(mac_cready, 0.0, 0.0).speed_ms * 3.6)
            == speed_to_fly
        )


def test_speed_to_fly_datapoints():
    polar = get_polar("LS 10 18m")
    assert polar is not None

    polar = polar.with_mass(540.0)
    assert round(polar.coeffs[0], 6) == -0.001486
    assert round(polar.coeffs[1], 4) == 0.0776
    assert round(polar.coeffs[2], 3) == -1.633

    data = [(0.0, 119), (0.5, 136), (1.0, 152), (2.0, 178), (3.0, 201), (4.0, 222)]
    for mac_cready, speed_to_fly in data:
        found = polar.speed_to_fly(mac_cready, 0.0, 0.0).speed_ms * 3.6
        assert round(found) == speed_to_fly


def test_bugs():
    polar = get_polar("LS 10 18m")
    assert polar is not None

    polar = polar.with_mass(540.0)

    polar.coeffs = add_bugs(polar.coeffs, 0.05)
    assert round(polar.coeffs[0], 6) == -0.00156
    assert round(polar.coeffs[1], 4) == 0.0815
    assert round(polar.coeffs[2], 3) == -1.715

    # the speeds should be slower
    data = [(0.0, 119), (0.5, 136), (1.0, 150), (2.0, 176), (3.0, 198), (4.0, 218)]
    for mac_cready, speed_to_fly in data:
        found = polar.speed_to_fly(mac_cready, 0.0, 0.0).speed_ms * 3.6
        assert round(found) == speed_to_fly
