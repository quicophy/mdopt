import json
import itertools
from itertools import permutations
import numpy as np
import copy

'''
All params lists must be in lists formats.
'''


def generate_all_batches_lists(large_file):
    gigalist = []
    loop = 0
    for a_list in large_file:
        for parameter in large_file[a_list]:
            dump = []
            if loop == 0:
                gigalist = large_file[a_list][parameter]
            elif loop == 1:
                for x in gigalist:
                    for y in large_file[a_list][parameter]:
                        dump.append([x, y])
                gigalist = dump
            else:
                for x in gigalist:
                    for y in large_file[a_list][parameter]:
                        dump.append([x+[y]][0])
                gigalist = dump
            loop += 1

    return gigalist


def gen_dict_from_lists(large_file, params_lists):
    full_dicts = []
    for par_list in params_lists:
        dump = copy.copy(large_file)
        iter = 0
        for sub_dic in dump:
            for dict_param in dump[sub_dic]:
                dump[sub_dic][dict_param] = par_list[iter]
                iter += 1
        full_dicts.append(dump)

    return full_dicts


def create_params_dict_list(file_from='batch_params.json', file_to='param_dict_list.json'):
    with open(file_from) as myfile:
        params = json.load(myfile)

    # Generate the list of dictionnary
    params_lists = generate_all_batches_lists(params)
    full_dict = gen_dict_from_lists(params, params_lists)

    # Overwrite previous json file with same name and closes it
    out_file = open(file_to, "w")
    json.dump(full_dict, out_file)
    out_file.close()


create_params_dict_list()
