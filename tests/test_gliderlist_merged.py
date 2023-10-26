import csv
from src.plot import get_all_polars


def test_data_for_wing_area():
    """If wingarea is provided, min and max weight also need to be there"""
    with open("data/gliderlist_merged.csv") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        for row in reader:
            if not row[6]:
                continue

            wing_area = float(row[6])
            assert 6.0 <= wing_area <= 25.0, wing_area

            min_weight = float(row[7])
            assert 60.0 <= min_weight <= 750.0, row

            mtow = float(row[9])
            assert 190.0 <= mtow <= 980.0


def test_data_for_polar():
    """If polar data is provided, we also need ref weight and min speed"""
    with open("data/gliderlist_merged.csv") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        for row in reader:
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
            assert 100.0 <= min_weight <= 750.0

            # mass
            mass = float(row[8])
            assert 150.0 <= mass <= 950.0

            # mtow
            mtow = float(row[9])
            assert 200.0 <= mtow <= 980.0

            # min speed
            min_speed = float(row[10])
            assert 45.0 <= min_speed <= 95.0, row


def test_speed_to_fly_club():
    polars = get_all_polars(comp_class="Club")
    assert len(polars) == 103

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
