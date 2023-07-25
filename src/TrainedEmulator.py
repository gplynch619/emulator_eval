import tensorflow as tf
import numpy as np
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import matplotlib
import pickle as pkl
import sys
import os

class TrainedEmulator:
    """ A basic class to unpack trained CONNECT Emulators and examine their products.
    """
    def __init__(self, path):
        self.path = path
        self.model = tf.keras.models.load_model(self.path, compile=False)

        with open(self.path + '/output_info.pkl', 'rb') as f:
            self.output_info = pkl.load(f)

        try:
            if self.output_info['normalize']['method'] == 'standardization':
                self.normalize = 'standardization'
            elif self.output_info['normalize']['method'] == 'log':
                self.normalize = 'log'
            elif self.output_info['normalize']['method'] == 'min-max':
                self.normalize = 'min-max'
            elif self.output_info['normalize']['method'] == 'factor':
                self.normalize = 'factor'
            else:
                self.normalize = 'factor'
        except:
            self.normalize = 'standardization'
            
        self.ell = self.output_info["ell"]
        #non_cl_num = 0
        #if "output_z" in self.output_info:
        #    non_cl_num += len(self.output_info["output_z"])
        #if "output_derived" in self.output_info:
        #    non_cl_num += len(self.output_info["output_derived"])
        #cl_idx = len(self.output_info["output_Cl"]) - non_cl_num
        #print(cl_idx)
        #self.output_info["output_Cl"] = self.output_info["output_Cl"][:cl_idx]
        self.available_products = []
        self.available_products += self.output_info["output_Cl"]
        if "sigma8" in self.output_info["output_z"]:
            self.output_info["output_z"][self.output_info["output_z"].index("sigma8")] = "sigma8_z"
        if "output_z" in self.output_info:
            self.available_products += self.output_info["output_z"]
        if "output_derived" in self.output_info:
            self.available_products += self.output_info["output_derived"]

    def get_predictions(self, input_models):

        v = tf.constant(input_models)
        eq_predict = self.model(v).numpy()
        if self.normalize == 'standardization':
            mean = self.output_info['normalize']['mean']
            var  = self.output_info['normalize']['variance']
            eq_predict = eq_predict * np.sqrt(var) + mean
        elif self.normalize == 'min-max':
            x_min = np.array(self.output_info['normalize']['x_min'])
            x_max = np.array(self.output_info['normalize']['x_max'])
            eq_predict = eq_predict * (x_max - x_min) + x_min

        return eq_predict
     
    def get_output_indices(self, product):
        if product in self.output_info["output_Cl"]:
            lim0 = self.output_info['interval']["Cl"][product][0]
            lim1 = self.output_info['interval']["Cl"][product][1]
        elif ("output_z" in self.output_info.keys() and product in self.output_info["output_z"]):
            if product == "sigma8_z":
                name = "sigma8"
            else:
                name = product
            lim0 = self.output_info['interval']["z_func"][name][0]
            lim1 = self.output_info['interval']["z_func"][name][1]
        elif ("output_derived" in self.output_info.keys() and product in self.output_info["output_derived"]):
            lim1 = self.output_info['interval']["derived"][product]
            lim0 = lim1-1
        
        return lim0, lim1

            

    def get_predictions_dict(self, input_model_dict):
        in_model_list = []

        for param in self.output_info["input_names"]:
            in_model_list.append(input_model_dict[param])
        
        input_models = np.vstack(in_model_list).T

        predictions = self.get_predictions(input_models)

        out_dict = {}
        for product in self.available_products:
            lim0, lim1 = self.get_output_indices(product)
            
            if ("output_derived" in self.output_info.keys() and product in self.output_info["output_derived"]):     
                out_dict[product] = predictions[:, lim0:lim1].flatten()
            else:
                out_dict[product] = predictions[:, lim0:lim1]

        return out_dict
    
    def get_error(self, units="cv"):

        with open(self.path + '/test_data.pkl', 'rb') as f:
            test_data = pkl.load(f)

        try:
            model_params = test_data[0]
            eq_data     = test_data[1]
        except:
            test_data = tuple(zip(*test_data))
            model_params = np.array(test_data[0])
            eq_data     = np.array(test_data[1])

        if self.normalize == 'standardization':
            mean = self.output_info['normalize']['mean']
            var  = self.output_info['normalize']['variance']
            eq_data = eq_data * np.sqrt(var) + mean
        elif self.normalize == 'min-max':
            x_min = np.array(self.output_info['normalize']['x_min'])
            x_max = np.array(self.output_info['normalize']['x_max'])
            eq_data = eq_data * (x_max - x_min) + x_min

        model_param_dict = {}
        for i,param in enumerate(self.output_info["input_names"]):
            model_param_dict[param] = model_params[:, i]

        predictions = self.get_predictions_dict(model_param_dict)

        errors = {}
        for product in self.available_products:
            if product=="tt" or product=="ee":
                lim0, lim1 = self.get_output_indices(product)
                if units=="cv":
                    norm = np.sqrt(2/(2*self.ell+1))*np.sqrt(np.array(eq_data[:, lim0:lim1])**2)
                elif units=="percent":
                    norm = np.sqrt(np.array(eq_data[:, lim0:lim1])**2)
                err = (predictions[product] - eq_data[:, lim0:lim1]) / norm
            elif product=="te":
                lim0, lim1 = self.get_output_indices("tt")
                tt_true = np.array(eq_data[:,lim0:lim1])
                lim0, lim1 = self.get_output_indices("ee")
                ee_true = np.array(eq_data[:, lim0:lim1])
                lim0, lim1 = self.get_output_indices("te")
                te_true = np.array(eq_data[:, lim0:lim1])
                if units=="cv":
                    norm = np.sqrt(1./(2*self.ell + 1))*np.sqrt(te_true**2 + tt_true*ee_true)
                elif units=="percent":
                    norm = np.sqrt(tt_true*ee_true)
                err = (predictions[product] - te_true) / norm
            else:
                lim0, lim1 = self.get_output_indices(product)
                err = (predictions[product] - eq_data[:, lim0:lim1]) / np.sqrt( np.array(eq_data[:, lim0:lim1])**2 )
            
            errors[product] = np.abs(err)

        return errors
