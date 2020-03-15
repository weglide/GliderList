#! python3.7
import csv
import exrex
import os.path

SIMPLELIST = 'indexlist.csv'
EXTENDEDLIST = 'extended_indexlist.csv'

class ListBuilder:
    def __init__(self):
        self.simple_gliders = []
        self.imported_extended = []
        self.created_extended = []

    def get_list(self, kind = 0):
        if kind == 0:
            self.simple_gliders = read_list(SIMPLELIST, True)
        else:
            self.imported_extended = read_list(EXTENDEDLIST, False)

    def create_extended_dict(self):
        for glider in self.simple_gliders:
            variants = list(exrex.generate(glider['Models']))
            for variant in variants:
                modified_glider = {}
                for key in glider:
                    modified_glider[key] = glider[key]
                modified_glider['Models'] = variant
                self.created_extended.append(modified_glider)

    def extend_extended(self):
        index_lookup = [d['Models'] for d in self.imported_extended]
        for glider in self.created_extended:
            if not glider['Models'] in index_lookup:
                self.imported_extended.append(add_columns(glider))

            else:
                index = index_lookup.index(glider['Models'])
                for key in glider.keys():
                    if key not in self.imported_extended[index].keys() and key.isdigit():
                        if glider[key] == '':
                            self.imported_extended[index][key] = get_last_index(glider, key)
                        else:
                            self.imported_extended[index][key] = glider[key]
 kl
    def save_extended_dict(self):
        with open(EXTENDEDLIST, mode = 'w') as writefile:
            corrected = []
            for glider in self.created_extended:
                corrected.append(add_columns(glider))

            fieldnames = corrected[0].keys()
            writer = csv.DictWriter(writefile, fieldnames=fieldnames)
            writer.writeheader()
            for glider in corrected:
                writer.writerow(glider)

    def update_extended_dict(self):
        print('updates')
        with open(EXTENDEDLIST, mode = 'w') as writefile:
            fieldnames = self.imported_extended[0].keys()
            writer = csv.DictWriter(writefile, fieldnames=fieldnames)
            writer.writeheader()
            for glider in self.imported_extended:
                writer.writerow(glider)


def get_last_index(glider, key):
    if str(int(key)-1) not in glider.keys():
        return ''
    if glider[str(int(key)-1)] != '':
        return glider[str(int(key)-1)]
    else:
        return get_last_index(glider, str(int(key)-1))

def escape(model):
    if '\.' in model:
        return model
    else:
        return model.replace('.', '\.')

def read_list(path, should_escape):
    with open(path, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        gliders = []
        for row in csv_reader:
            glider = {}
            for item in row.items():
                if should_escape:
                    if item[0] == 'Models':
                        glider[item[0]] = escape(item[1])
                    else:
                        glider[item[0]] = item[1]
                else:
                    glider[item[0]] = item[1]

            gliders.append(glider)
    return gliders

insert = lambda _dict, obj, pos: {k: v for k, v in (list(_dict.items())[:pos] + list(obj.items()) + list(_dict.items())[pos:])}

def add_columns(glider):
    required_columns = ['Glider', 'Models', 'Manufacturer', 'Competition Class', 'Engine', 'Double Seater', 'Winglets']
    for key in required_columns:
        if key not in glider.keys():
            glider = insert(glider, {key:''}, 4)
            #glider[key] = ''
    return glider

if __name__ == "__main__":
    list_builder = ListBuilder()
    list_builder.get_list(0)
    list_builder.create_extended_dict()

    if not os.path.isfile(EXTENDEDLIST):
        list_builder.save_extended_dict()
    else:
        list_builder.get_list(1)
        list_builder.extend_extended()
        list_builder.update_extended_dict()
