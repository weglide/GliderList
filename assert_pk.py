import csv


def assert_pk():
    """Assert no duplicates in pk"""
    with open("extended.csv") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        pks = [row[0] for row in reader]
    assert len(pks) == len(set(pks))


if __name__ == "__main__":
    assert_pk()
