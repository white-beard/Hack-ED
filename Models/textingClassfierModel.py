from tensorflow.keras.models import model_from_json
import numpy as np


class Texting_VS_Normal(object):

    Activity_LIST = ["Texting", "Normal"]

    def __init__(self, texting_vs_normal_neural_net, texting_vs_normal_neural_net_weights_h5):
        """ Loading our Model architecture and Weights """
        with open(texting_vs_normal_neural_net, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)
        self.loaded_model.load_weights(texting_vs_normal_neural_net_weights_h5)


    def predict(self,img):
        self.preds = self.loaded_model.predict(np.array([img]))
        self.preds_proba = self.loaded_model.predict_proba(np.array([img]))
        if self.preds >= 0.1:
            self.verdict = "Texting"
        else:
            self.verdict = "Normal"
        return self.verdict, self.preds_proba
