import numpy as np
import classy as Class
import sys
import pickle

settings_dict_filename = sys.argv[1]
models_filename = sys.argv[2]
outfiles_filename = sys.argv[3]

with open(settings_dict_filename, 'rb') as f:
    settings= pickle.load(f)
with open(outfiles_filename, 'rb') as f:
    outfiles = pickle.load(f)
with open(models_filename, 'rb') as f:
    model = pickle.load(f)

output_spectra = list(outfiles.keys())
ll_max = settings["l_max_scalars"]
M = Class.Class()
M.set(settings)

try:
    M.compute()
    for xx in output_spectra:
        spectrum = M.lensed_cl(ll_max)[xx][2:]
        out_array = np.hstack((model, spectrum))
        with open(outfiles[xx], 'ab') as f:
           np.savetxt(f, [out_array])

        ### debugging
        #with open("log_{}.txt".format(rank), "a") as f:
        #    process = psutil.Process(os.getpid())
        #    f.write("EB size: {0} KB \t Total mem used: {1} KB\n".format(sys.getsizeof(M)/1000., process.memory_info().rss/1000))
            ### end debugging
    M.struct_cleanup()
    M.empty()

except Class.CosmoComputationError as failure_message:
    print("Mode failed ")
    print(str(failure_message)+'\n')
    M.struct_cleanup()
    M.empty() 
except Class.CosmoSevereError as critical_message:
    print("Something went wrong when calling CLASS" + str(critical_message))
    M.struct_cleanup()
    M.empty()