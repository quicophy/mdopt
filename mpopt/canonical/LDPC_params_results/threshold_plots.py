import matplotlib.pyplot as plt
from intersect import intersection
import numpy as np


# Import result file and get all curve points

def import_data_file(path_to_file='data.dat', header=True):
    '''
    Reads the data saved by the save_to_file function. Returns a numpy array
    without fixed variable types.
    '''
    names = True
    if header is False:
        names = None
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


def gather(params, data):
    kept = []
    for line in data:
        keep=True
        for param in params:
            included=False
            for val in params[param]:
                if val== line[param]:
                    included=True
            if included==False:
                keep = False
        if keep==True:
            kept.append(line)
    return kept


def return_points(params, data):
    params_val_lists = [0 for _ in range(len(params))]
    print(str(len(plot_lists))+' list lenght')
    for line in data:
        for param in params:
            print(None)
            #if line[params[param]]:

def no_correction(check_degree=4,size_mult=1,start=0,end=0.2):
    lin_space = np.linspace(start,end,200)
    bit_num = size_mult*check_degree


#Important params
params_select = {
    'code_size_mult': [3],
    'chi_max': [False]}

data = import_data_file(path_to_file='results_list.txt')
keeps = gather(params_select, data)

#params_to_plot = ['bit_degree','check_degree', 'code_size_mult','chi_max','phys_err_rt']
phys_err_rt = []
failure_rt = []
fail_std = []
avg_time = []
time_std = []

for line in keeps:
    print(line)
    phys_err_rt.append(line[3])
    failure_rt.append(line[16])
    fail_std.append(line[17])   
    avg_time.append(line[18])
    time_std.append(line[19])

plt.errorbar(phys_err_rt, failure_rt, yerr=fail_std)
#plt.plot(phys_err_rt, failure_rt, y_err=fail_std)
plt.show()


#'failure_rt', 'fail_std', 'avg_time','time_std'

#points = return_points(params_to_plot ,keeps)


# Create curve arrays for each param set

# Create and compute no-correction error probability curve


# Find intersection point for all of these using intersect, plus error on point.


# Plot semi-trheshold curve for all sizes to 1/n_bits and do a fit to find real threshold value
