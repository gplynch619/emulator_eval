import sys
import os
import numpy as np
from importlib.machinery import SourceFileLoader
from cosmopower import cosmopower_NN
import tensorflow as tf

device='gpu:0' if tf.test.is_gpu_available() else 'cpu:0'
print('using', device, 'device \n')

param = SourceFileLoader(sys.argv[1], sys.argv[1])

data_path = os.path.join(param.basepath, "data")

training_parameters_file = os.path.join(data_path, param.parameter_file)
training_data_file = os.path.join(data_ath, param.data_file)

training_parameters = np.load(training_parameters_file)
training_data = np.load(training_data_file)

training_spectra = training_data['features']
ell_range = training_data['modes']

model_parameters=param.parameter_names

cp_nn = cosmopower_NN(parameters=model_parameters,
                      modes=ell_range,
                      n_hidden=param.network_architecture,
                      verbose=True)

save_model_path = os.path.join(basepath, "models", param.save_name)

with tf.device(device):
    cp_nn.train(training_parameters=training_parameters,
                training_featurers=training_spectra,
                filename_saved_model=save_model_path,
                validation_split=param.validation_split,
                learning_rates=param.learning_rates,
                batch_sizes=param.batch_sizes,
                gradient_accumulation_steps=param.gradient_accumulation_steps,
                patience_values=param.patience_values,
                max_epochs=param.max_epochs
                )
