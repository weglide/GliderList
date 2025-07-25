import csv

import numpy as np
import pytest
from polar import Polar


@pytest.fixture(scope="session")
def gliderlist() -> list[list[str]]:
    with open("gliderlist.csv") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        return [row for row in reader]


def test_unique_pk(gliderlist: list[list[str]]):
    pks = [row[0] for row in gliderlist]
    assert len(pks) == len(set(pks))


def test_unique_name(gliderlist: list[list[str]]):
    names = [row[2] for row in gliderlist]
    assert len(names) == len(set(names))


def test_self_launcher_is_motorglider(gliderlist: list[list[str]]):
    for row in gliderlist:
        if row[15] == "x":
            assert row[5] != "GL"


def test_double_is_doubleseater(gliderlist: list[list[str]]):
    for row in gliderlist:
        if row[4] == "Double":
            assert row[12] == "x"


def test_data_for_wing_area(gliderlist: list[list[str]]):
    """If wingarea is provided, min and max weight also need to be there"""
    for row in gliderlist:
        if not row[6]:
            continue

        wing_area = float(row[6])
        assert 6.0 <= wing_area <= 25.0, wing_area

        min_weight = float(row[7])
        assert 60.0 <= min_weight <= 750.0, row

        mtow = float(row[9])
        assert 155.0 <= mtow <= 980.0


def test_data_for_polar(gliderlist: list[list[str]]):
    """If polar data is provided, we also need ref weight and min speed"""
    for row in gliderlist:
        data = row[11].split(":")
        if not data or not data[0]:
            continue

        coeffs = [float(data[i]) for i in range(0, len(data))]
        assert len(coeffs) == 3

        # wing area
        wing_area = float(row[6])
        assert 6.0 <= wing_area <= 22.0, wing_area

        # min weight
        min_weight = float(row[7])
        assert 60.0 <= min_weight <= 750.0

        # mass
        mass = float(row[8])
        assert 150.0 <= mass <= 950.0

        # mtow
        mtow = float(row[9])
        assert 191.0 <= mtow <= 980.0

        # min speed
        min_speed = float(row[10])
        assert 45.0 <= min_speed <= 95.0, row


def all_polars(
    gliderlist: list[list[str]], comp_class: str | None = None
) -> list[Polar]:
    polars = []
    for row in gliderlist:
        data = row[11].split(":")
        if not data or not data[0]:
            continue

        if comp_class is not None and row[4] != comp_class:
            continue

        coeffs = np.array([float(data[i]) for i in range(0, len(data))])
        mass = float(row[8])
        min_speed_ms = float(row[10]) / 3.6
        mtow = float(row[9])
        polar = Polar(coeffs, mass, min_speed_ms, row[2], mtow, wing_area=float(row[6]))
        polars.append(polar)
    return polars


def test_polar_numerics_club_class(gliderlist: list[list[str]]):
    polars = all_polars(gliderlist, comp_class="Club")
    assert len(polars) > 100

    for polar in polars:
        ignore = ["Phöbus", "Silent", "mini LAK", "AK-5"]
        if any([i in polar.name for i in ignore]):
            continue

        # test speed to fly for macCready of two
        speed_to_fly = polar.speed_to_fly(2.0, 0.0, 0.0).speed_ms * 3.6
        assert 80.0 <= speed_to_fly <= 150.0, polar.name

        # test best ld value
        assert 20.0 <= polar.best_ld_value <= 43.0, polar.name

        # test best ld speed
        assert 70.0 <= (polar.best_ld.speed_ms * 3.6) <= 115.0, polar.name


def test_polar_numerics_double(gliderlist: list[list[str]]):
    polars = all_polars(gliderlist, comp_class="Double")
    assert len(polars) > 90

    for polar in polars:
        # test speed to fly for macCready of two
        speed_to_fly = polar.speed_to_fly(2.0, 0.0, 0.0).speed_ms * 3.6
        assert 80.0 <= speed_to_fly <= 180.0, polar.name

        # test best ld value
        assert 15.0 <= polar.best_ld_value <= 60.0, polar.name

        # test best ld speed
        assert 60.0 <= (polar.best_ld.speed_ms * 3.6) <= 140.0, polar.name


def test_polar_numerics_15m(gliderlist: list[list[str]]):
    polars = all_polars(gliderlist, comp_class="15")
    assert len(polars) > 70

    for polar in polars:
        # test speed to fly for macCready of two
        speed_to_fly = polar.speed_to_fly(2.0, 0.0, 0.0).speed_ms * 3.6
        assert 140.0 <= speed_to_fly <= 200.0, polar.name

        # test best ld value
        assert 35.0 <= polar.best_ld_value <= 52.0, polar.name

        # test best ld speed
        assert 85.0 <= (polar.best_ld.speed_ms * 3.6) <= 125.0, polar.name


def test_polar_numerics_18m(gliderlist: list[list[str]]):
    polars = all_polars(gliderlist, comp_class="18")
    assert len(polars) > 80

    for polar in polars:
        # test speed to fly for macCready of two
        speed_to_fly = polar.speed_to_fly(2.0, 0.0, 0.0).speed_ms * 3.6
        assert 130.0 <= speed_to_fly <= 200.0, polar.name

        # test best ld value
        assert 40.0 <= polar.best_ld_value <= 58.0, polar.name

        # test best ld speed
        assert 85.0 <= (polar.best_ld.speed_ms * 3.6) <= 135.0, polar.name


def test_polar_numerics_open(gliderlist: list[list[str]]):
    polars = all_polars(gliderlist, comp_class="Open")
    assert len(polars) > 75

    for polar in polars:
        # test speed to fly for macCready of two
        speed_to_fly = polar.speed_to_fly(2.0, 0.0, 0.0).speed_ms * 3.6
        assert 130.0 <= speed_to_fly <= 200.0, polar.name

        # test best ld value
        assert 38.0 <= polar.best_ld_value <= 70.0, polar.name

        # test best ld speed
        assert 80.0 <= (polar.best_ld.speed_ms * 3.6) <= 130.0, polar.name


def test_polar_numerics_standard(gliderlist: list[list[str]]):
    polars = all_polars(gliderlist, comp_class="Standard")
    assert len(polars) > 45

    for polar in polars:
        # test speed to fly for macCready of two
        speed_to_fly = polar.speed_to_fly(2.0, 0.0, 0.0).speed_ms * 3.6
        assert 130.0 <= speed_to_fly <= 160.0, polar.name

        # test best ld value
        assert 40.0 <= polar.best_ld_value <= 50.0, polar.name

        # test best ld speed
        assert 90.0 <= (polar.best_ld.speed_ms * 3.6) <= 115.0, polar.name
