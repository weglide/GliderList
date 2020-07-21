import exrex
import csv
import os.path
import datetime
from copy import copy

SIMPLELIST = 'indexlist.csv'
EXTENDEDLIST = 'extended_indexlist.csv'
CHANGELOG = 'CHANGELOG.md'


def escapeit(model):
    if '\.' in model:
        return model
    else:
        return model.replace('.', '\.')


def read_list(path) -> list:
    gliders = []
    with open(path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            glider = {}
            for glider_info in row:
                if glider_info == 'Models':
                    glider[glider_info] = escapeit(row[glider_info])
                else:
                    glider[glider_info] = row[glider_info]
            gliders.append(glider)
    return gliders


class Extender:
    def __init__(self):
        self.simple_list = []
        self.simple_extended = []
        self.extended_list = []
        self.removed = []
        self.added = []

    def import_lists(self, first: bool = False) -> None:
        self.simple_list = read_list(SIMPLELIST)
        if os.path.isfile(EXTENDEDLIST):
            self.extended_list = read_list(EXTENDEDLIST)

    def extend_simple_list(self) -> None:
        for glider in self.simple_list:
            variants = list(exrex.generate(glider['Models']))
            for variant in variants:
                generated_glider = copy(glider)
                del generated_glider['Models']
                generated_glider['Model'] = variant
                self.simple_extended.append(generated_glider)

    def merge_lists(self) -> None:
        # delete removed gliders
        new_extended = []
        for glider in self.extended_list:
            already_merged = [item for item in self.simple_extended if item.get(
                'Model') == glider['Model']]
            if len(already_merged) != 0:
                new_extended.append(glider)
            else:
                self.removed.append(glider['Model'])
        self.extended_list = copy(new_extended)

        # Insert new gliders from simple_list
        for glider in self.simple_extended:
            already_merged = [item for item in self.extended_list if item.get(
                'Model') == glider['Model']]
            # Only add glider if not already in extended list
            if len(already_merged) == 0:
                self.added.append(glider['Model'])
                glider['Winglets'] = ''
                glider['Double Seater'] = ''
                glider['FES'] = ''
                glider['Engine'] = ''
                self.extended_list.append(glider)

    def sort_extended(self) -> None:
        self.extended_list.sort(key=lambda x: x['Manufacturer'])
        self.extended_list.sort(key=lambda x: x['Model'])
        self.extended_list.sort(
            key=lambda x: x['2020'], reverse=True)
        self.extended_list.sort(
            key=lambda x: x['Competition Class'])

    def save_extended(self) -> None:
        with open(EXTENDEDLIST, 'w') as csvfile:
            fieldnames = self.extended_list[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for glider in self.extended_list:
                writer.writerow(glider)

    def write_changelog(self) -> None:
        with open(CHANGELOG, 'r') as changelog_file:
            content = changelog_file.read()

        with open(CHANGELOG, 'w') as changelog_file:
            changelog_file.truncate(0)
            now = datetime.datetime.now()
            changelog_file.write(f'## {now:%Y-%m-%d}\n')
            changelog_file.write('### Added\n')
            for glider in self.added:
                changelog_file.write(f'- {glider}\n')
            changelog_file.write('### Removed\n')
            for glider in self.removed:
                changelog_file.write(f'- {glider}\n')
            changelog_file.write('---\n')
            changelog_file.write(content)


extender = Extender()
extender.import_lists()
extender.extend_simple_list()
extender.merge_lists()
extender.sort_extended()
extender.save_extended()
extender.write_changelog()
