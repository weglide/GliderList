import numpy as np
import csv
from .polar import open_polar, Polar
from typing import NamedTuple


class XCSoarData(NamedTuple):
    name: str
    ref_mass: float
    empty_mass: float
    wing_area: float

    speed1: float
    sink1: float
    speed2: float
    sink2: float
    speed3: float
    sink3: float

    @property
    def polar_string(self) -> str:
        return f"{self.speed1}:{self.sink1}:{self.speed2}:{self.sink2}:{self.speed3}:{self.sink3}"


def open_gliderlist():
    with open("gliderlist.csv") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        gliders = []
        for row in reader:
            gliders.append(row)

        print(f"Parsed {len(gliders)} gliders from gliderlist")
    return gliders


def open_gliderlist_data() -> dict[str, list]:
    with open("data/gliderlist_data.csv") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        gliders = {}
        for row in reader:
            gliders[row[0]] = row[3:7]
            polar = None
            if row[7].endswith((".POL", ".pol")):
                reference_mass = float(row[5])
                polar = Polar.from_filename(row[7], reference_mass)
            elif ":" in row[7]:
                # three speeds and sink, do polyfit
                data = row[7].split(":")
                points = [
                    (float(data[i]), float(data[i + 1])) for i in range(0, len(data), 2)
                ]
                if row[8]:
                    min_speed_ms = float(row[8]) / 3.6
                else:
                    min_speed_ms = (points[0][0] - 20) / 3.6
                polar = Polar.from_data_points(
                    points, float(row[5]), min_speed_ms, filename="", order=2
                )
            elif "=" in row[7]:
                # coeffs directly in file
                coeffs = np.array([float(r) for r in row[7].split("=")])
                polar = Polar(
                    coeffs=coeffs,
                    mass=float(row[5]),
                    min_speed_ms=float(row[8]) / 3.6,
                    name="",
                )

            if polar is not None:
                sink = polar.evaluate(40, 0, 0)
                print(f"Sink at 144 km/h for {row[2]}: {sink:.3f}")
                if sink > 0:
                    raise AssertionError

            polar_data = (
                ["", ""]
                if polar is None
                else [
                    f"{polar.min_speed_ms * 3.6:.2f}",
                    ":".join([f"{c:.7f}" for c in polar.coeffs]),
                ]
            )
            gliders[row[0]].extend(polar_data)
        print(f"Parsed {len(gliders)} gliders from data gliderlist")
    return gliders


def open_xcsoar() -> dict[str, XCSoarData]:
    # ID,Name,ReferenceMass,Maxwater,Speed1,Sink1,Speed2,Sink2,Speed3,Sink3,Wingarea,NoIdea,Index,Empty Mass
    with open("data/xcsoar.csv") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        gliders = {}
        for row in reader:
            if not row[0]:
                continue
            data = XCSoarData(
                name=row[1],
                ref_mass=float(row[2]),
                wing_area=float(row[10]),
                empty_mass=float(row[13]),
                speed1=float(row[4]),
                sink1=float(row[5]),
                speed2=float(row[6]),
                sink2=float(row[7]),
                speed3=float(row[8]),
                sink3=float(row[9]),
            )
            gliders[row[0]] = data

        print(f"Parsed {len(gliders)} gliders from xcsoar")
    return gliders


def open_polars():
    with open("data/polars.csv") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        gliders = {}
        for row in reader:
            if not row[0]:
                continue
            id = row[0]
            polar_file = row[2]
            mass_90kg = float(row[5])
            mtow = float(row[6])
            try:
                polar_info, polar_data = open_polar(polar_file)
            except FileNotFoundError:
                print(f"File {polar_file} not found")
                continue
            gliders[id] = [
                polar_info.wingarea,
                mass_90kg - 90,
                polar_info.mass,
                mtow,
                polar_file,
            ]
    print(f"Parsed {len(gliders)} gliders from polar file")
    return gliders


def write(gliders, polars: dict[int, list], xcsoar_data: dict[int, XCSoarData]):
    with open("data/dmst_new.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "ID",
                "Glider",
                "Model",
                "Wingarea",
                "Empty Mass",
                "Reference Mass",
                "MTOW",
                "Polar",
            ]
        )
        for glider in gliders:
            if (data := polars.get(glider[0])) is not None:
                writer.writerow(glider[:3] + data)
            elif (data := xcsoar_data.get(glider[0])) is not None:
                writer.writerow(
                    glider[:3]
                    + [
                        data.wing_area,
                        data.empty_mass,
                        data.ref_mass,
                        "",
                        data.polar_string,
                    ]
                )
            else:
                writer.writerow(glider[:3] + [""] * 5)


def merge(gliders: list, gliders_data: dict):
    with open("data/gliderlist_merged.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                # [0: 6] glider
                "ID",
                "Glider",
                "Model",
                "Manufacturer",
                "Competition Class",
                "Kind",
                # [6: 10] data
                "Wingarea",
                "Empty Weight",
                "Reference Mass",
                "MTOW",
                # [10: 12] polar data
                "Min Speed",
                "Polar Coeffs",
                # [12: ] glider
                "Double Seater",
                "Exclude Live",
                "Vintage",
                "Self-Launcher",
                "2016",
                "2017",
                "2018",
                "2019",
                "2020",
                "2021",
                "2022",
                "2023",
                "2024",
            ]
        )
        for glider in gliders:
            data = gliders_data.get(glider[0])
            try:
                writer.writerow(glider[:6] + data + glider[6:7] + glider[8:])
            except TypeError as e:
                print(glider, e)


if __name__ == "__main__":
    gliders = open_gliderlist()
    glider_data = open_gliderlist_data()
    merge(gliders, glider_data)
