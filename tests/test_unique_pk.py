import csv


def test_unique_pk():
    with open("gliderlist.csv") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        pks = [row[0] for row in reader]
    assert len(pks) == len(set(pks))


def test_unique_name():
    with open("gliderlist.csv") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        pks = [row[2] for row in reader]
    assert len(pks) == len(set(pks))


if __name__ == "__main__":
    test_unique_pk()
