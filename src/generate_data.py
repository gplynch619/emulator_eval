import numpy as np
from importlib.machinery import SourceFileLoader
import sys
import os
import re
import time
from mpi4py import MPI
from scipy.stats import qmc
import itertools
import classy as Class

######################################################
#  Read parameters
######################################################
param_file = sys.argv[1]
param = SourceFileLoader(param_file, param_file).load_module()

input_param_ranges = param.param_ranges
additional_settings = param.additional_settings
output_spectra = param.output_spectra
ll_max = param.ll_max

######################################################
#  Input parsing
######################################################

def parse_input(input_ranges):

    if("xe_control_points" in input_ranges.keys()):
        modified_rec = True
    else:
        modified_rec = False

    if modified_rec:
        required_settings = ["xe_pert_type", "xe_control_pivots", "zmin_pert", "zmax_pert", "xe_pert_num"]
        for setting in required_settings:
            assert setting in additional_settings.keys(), "Specify {0} in additional settings before continuing".format(setting)
        param_ranges = input_ranges.copy()
        cp_lhc_range = param_ranges["xe_control_points"]
        param_ranges.pop("xe_control_points")
        for i in np.arange(1, additional_settings["xe_pert_num"]-1):
            name = "q_{}".format(i)
            param_ranges[name] = cp_lhc_range
    else:
        param_ranges = input_ranges
    
    return param_ranges

param_ranges = parse_input(input_param_ranges)

Nparams = len(param_ranges.keys())

N = param.N

######################################################
#  Create Latin Hypercube of samples
######################################################
def create_models(N, ranges):
    sampler = qmc.LatinHypercube(d=Nparams)
    samples = sampler.random(n=N)
    i=0
    for range in ranges.values():
        samples.T[i] *= range[1] - range[0]
        samples.T[i] += range[0]
        i+=1

    return samples

def create_name_mapping(param_names):
    mapping = {}
    i=0
    for name in param_names:
        mapping[i] = name
        i+=1
    return mapping

######################################################
#  Set up MPI
######################################################

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

Nworkers = size-1
next_worker = itertools.cycle(np.arange(1, size))
is_working = {}
for r in np.arange(size):
    is_working[str(r)] = False

pause = 0.1
short_pause = 1e-4
######################################################
#  Compute
######################################################


if rank==0:
    name_mapping = create_name_mapping(param_ranges.keys())
else:
    name_mapping = None

name_mapping = comm.bcast(name_mapping, root=0)

if rank==0:
    np.random.seed(1)
    data = create_models(N, param_ranges)
    #print(data)
    idx = 0
    while data.shape[0] > idx:
        target = next(next_worker)
        if comm.iprobe(target):
            _ = comm.recv(source=target)
            comm.send(data[idx], dest=target)
            idx+=1
            is_working[str(r)] = 1
        else:
            is_working[str(r)] = 0

        if all(value == 0 for value in is_working.values()):
            time.sleep(pause)
        else:
            time.sleep(short_pause)
        if(idx%1000==0):
            print("{} have been sent".format(idx))

    for i in np.arange(1, size):
        comm.send("done", dest=i)

if rank!=0:
    common_settings = {'output' : 'tCl,pCl,lCl',
                   'thermodynamics_verbose': 0,
                   'input_verbose': 0,
                   'lensing': 'yes',
                   'xe_pert_type': 'none'
                  }
    pr_cover_tau = 0.004
    precision_settings = {"start_sources_at_tau_c_over_tau_h": pr_cover_tau}
    M = Class.Class()

    outfiles = {}
    for xx in output_spectra:
        outfiles[xx] = open(os.path.join(param.outdir_root,"{}_spectrum.dat.{}".format(xx, rank)), "ab")
    
    while True:
        M.set(precision_settings)
        M.set(common_settings)
        
        if len(additional_settings)>0:
            M.set(additional_settings)
        
        comm.send("waiting for a model", dest=0)
        model = comm.recv(source=0)
        if type(model).__name__ == 'str': #breaks when receiving "done" signal
            break
        settings = {}
        control_points = []
        for i,param in enumerate(model):
            if("q_" in name_mapping[i]):
                control_points.append(model[i])
            else:
                settings[name_mapping[i]] = model[i]
        if len(control_points)>0:
            control_points = np.insert(control_points, 0, 0.0)
            control_points = np.append(control_points, 0.0)
            str_ctrl = [str(c) for c in control_points]
            settings["xe_control_points"] = ",".join(str_ctrl)

        M.set(settings)
        try:
            M.compute()
            for xx in output_spectra:
                spectrum = M.lensed_cl(ll_max)[xx][2:]
                out_array = np.hstack((model, spectrum))
                np.savetxt(outfiles[xx], [out_array])
            M.struct_cleanup()
            M.empty()
        except Class.CosmoComputationError as failure_message:
            print("Model {} failed ".format(model))
            print(str(failure_message)+'\n')
            M.struct_cleanup()
            M.empty() 
        except Class.CosmoSevereError as critical_message:
            print("Something went wrong when calling CLASS" + str(critical_message))
            M.struct_cleanup()
            M.empty()
    
    for f in outfiles.values():
        f.close()
    #done

MPI.Finalize()
sys.exit(0)
