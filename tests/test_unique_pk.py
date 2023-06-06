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


def test_selflauncher_is_motorglider():
    with open("gliderlist.csv") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        for row in reader:
            if row[10] == "x":
                assert row[5] != "GL"


def test_double_is_doubleseater():
    with open("gliderlist.csv") as file:
        reader = csv.reader(file, delimiter=",")
        next(reader)
        for row in reader:
            if row[4] == "Double":
                assert row[6] == "x"
