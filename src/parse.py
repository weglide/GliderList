import csv
from .polar import open_polar
from typing import NamedTuple, Dict, List


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


def open_gliderlist_data() -> Dict[str, List]:
    with open("data/gliderlist_data.csv") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        gliders = {}
        for row in reader:
            gliders[row[0]] = row[3:7]

        print(f"Parsed {len(gliders)} gliders from data gliderlist")
    return gliders


def open_xcsoar() -> Dict[int, XCSoarData]:
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


def write(gliders, polars: Dict[int, List], xcsoar_data: Dict[int, XCSoarData]):
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


def merge(gliders: List, gliders_data: Dict):
    with open("data/gliderlist_merged.csv", "w") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                # [0: 6]
                "ID",
                "Glider",
                "Model",
                "Manufacturer",
                "Competition Class",
                "Kind",
                # [6: 10]
                "Wingarea",
                "Empty Weight",
                "Reference Mass",
                "MTOW",
                # [10: ]
                "Double Seater",
                "Exclude Live",
                "Vintage",
                "2016",
                "2017",
                "2018",
                "2019",
                "2020",
                "2021",
                "2022",
                "2023",
            ]
        )
        for glider in gliders:
            data = gliders_data.get(glider[0])
            try:
                writer.writerow(glider[:6] + data + glider[6:7] + glider[8:])
            except TypeError:
                print(glider)


if __name__ == "__main__":
    gliders = open_gliderlist()
    glider_data = open_gliderlist_data()
    merge(gliders, glider_data)
