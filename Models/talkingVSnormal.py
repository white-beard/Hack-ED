from tensorflow.keras.models import model_from_json
import numpy as np

import tensorflow as tf

class Talking_vs_normal(object):

    Activity_LIST = ["Talking", "Normal"]

    def __init__(self, Talking_vs_normal_neural_net, Talking_vs_normal_neural_net_h5):
        # load model from JSON file
        with open(Talking_vs_normal_neural_net, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(Talking_vs_normal_neural_net_h5)
        #self.loaded_model._make_predict_function()

    def predict_activity(self,img):
        self.preds = self.loaded_model.predict(img)
        if self.preds >= 0.6:
            self.verdict = "Texting"
        else:
            self.verdict = "Normal"
        return self.verdict
