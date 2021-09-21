import numpy as np


def import_data_file(path_to_file='data.dat', header=True):
    '''
    Reads the data saved by the save_to_file function. Returns a numpy array
    without fixed variable types.
    '''
    names = True
    if header is False:
        names = False
    data = np.genfromtxt(path_to_file,
                         skip_header=0,
                         skip_footer=0,
                         names=names,
                         dtype=None,
                         delimiter=',',
                         deletechars=" !#$%&'()*+, -./:;<=>?@[\\]^{|}~",
                         autostrip=True,
                         replace_space='_')
    return data


def ploter_success_rates(**kwargs):
    for var in kwargs:
        for i in var:
            print(i)


#files = import_data_file()
ploter_success_rates(size=[0, 1, 2])
# print(files)
