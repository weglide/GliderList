import csv
import json

def test_ogn_ddb_mapping():
    with open("gliderlist.csv") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        gliders = {int(row[0]): row[1:] for row in reader}

    with open("ogn_ddb_mapping.json") as file:
        ogn_mapping = json.load(file)
        for v in ogn_mapping.values():
            if v:
                assert gliders.get(v["id"]) is not None, v


if __name__ == "__main__":
    test_ogn_ddb_mapping()
