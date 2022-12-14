import sys
import os
import numpy as np
from importlib.machinery import SourceFileLoader
from cosmopower import cosmopower_NN
import tensorflow as tf

device='gpu:0' if tf.test.is_gpu_available() else 'cpu:0'
print('using', device, 'device \n')

#paramfile = os.path.basename(sys.argv[1])
paramfile = sys.argv[1]
param = SourceFileLoader(paramfile, paramfile).load_module()

data_path = os.path.join(param.basepath, "data")

training_parameters_file = os.path.join(data_path, param.parameter_file)
training_data_file = os.path.join(data_path, param.data_file)

training_parameters = np.load(training_parameters_file)
training_data = np.load(training_data_file)

training_spectra = training_data['features']
ell_range = training_data['modes']
ell_range = np.hstack([ell_range, [2500]])

model_parameters=param.parameter_names

cp_nn = cosmopower_NN(parameters=model_parameters,
                      modes=ell_range,
                      n_hidden=param.network_architecture,
                      verbose=True)

save_model_path = os.path.join(param.basepath, "models", param.save_name)

print(ell_range)
print(training_spectra.shape)

with tf.device(device):
    cp_nn.train(training_parameters=training_parameters,
                training_features=training_spectra,
                filename_saved_model=save_model_path,
                validation_split=param.validation_split,
                learning_rates=param.learning_rates,
                batch_sizes=param.batch_sizes,
                gradient_accumulation_steps=param.gradient_accumulation_steps,
                patience_values=param.patience_values,
                max_epochs=param.max_epochs,
                )
