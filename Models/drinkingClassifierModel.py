from tensorflow.keras.models import model_from_json
import numpy as np

import tensorflow as tf

class Drinking_VS_Normal(object):
    Activity_LIST = ["Drinking", "Normal"]

    def __init__(self, Drinking_vs_normal_neural_net, Drinking_vs_normal_neural_net_h5):
        # load model from JSON file
        with open(Drinking_vs_normal_neural_net, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(Drinking_vs_normal_neural_net_h5)
        #self.loaded_model._make_predict_function()

    def predict(self, img):
        self.preds = self.loaded_model.predict(np.array([img]))
        self.preds_proba = self.loaded_model.predict_proba(np.array([img]))
        if self.preds >= 0.1:
            self.verdict = "Drinking"
        else:
            self.verdict = "Normal"
        return self.verdict, self.preds_proba

