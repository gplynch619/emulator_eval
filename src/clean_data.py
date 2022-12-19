import numpy as np
from importlib.machinery import SourceFileLoader
import sys
import os
import itertools
from generate_data import parse_input

######################################################
#  Read parameters
######################################################
param_file = sys.argv[1]
spectrum_file_dir = sys.argv[2]
param = SourceFileLoader(param_file, param_file).load_module()
param_ranges = parse_input(param.param_ranges)

param_names = list(param_ranges.keys())

output_spectra = param.output_spectra
ll_max = param.ll_max

modes = np.arange(2, ll_max+1)
print(modes)

Nparams = len(param_ranges.keys())
N = param.N

initial_files = os.listdir(spectrum_file_dir)

for spectrum_type in param.output_spectra:
    master_parameters = []
    master_spectra = []
    
    for filename in initial_files:
        if "{}_spectrum.dat".format(spectrum_type) in filename:
            spectra_and_params = np.loadtxt(os.path.join(spectrum_file_dir, filename))

            # clean NaN's if any
            rows = np.where(np.isfinite(spectra_and_params).all(1))
            spectra_and_params = spectra_and_params[rows]

            parameters = spectra_and_params[:, :Nparams]

            if spectrum_type=="tt":
                spectra = np.log10(spectra_and_params[:, Nparams:])
            else:
                spectra = spectra_and_params[:, Nparams:]
        

            master_parameters.append(parameters)
            master_spectra.append(spectra)
    
    master_parameters = np.vstack(master_parameters)
    master_spectra = np.vstack(master_spectra)
    parameters_dict = {param_names[i]: master_parameters[:, i] for i in range(len(param_names))}
    spectra_dict = {'modes': modes, 'features': master_spectra}
    np.savez(os.path.join(spectrum_file_dir, '{}_class_params.npz'.format(spectrum_type)), **parameters_dict)
    np.savez(os.path.join(spectrum_file_dir,'{}_class_logpower.npz'.format(spectrum_type)), **spectra_dict)

for filename in initial_files:
    os.remove(os.path.join(spectrum_file_dir, filename))