# Copyright 2019 David Kügler, Technical University of Darmstadt, Darmstadt
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from keras.models import load_model
from matplotlib import pyplot
import numpy
np=numpy

from predictorlayer import PredictorLayer

class PredictorKeras(PredictorLayer):
    """A class to perform the prediction of points given a keras model."""
    
    def __init__(self, model_state):
        """Initialize the predictor layer with a solver and a model state.

        :param model_state: path to the model / weights description of the network
                            The file should contain a full keras model with initialized weights.
                            keras_model.predict() must be callable on this object
        """
        
        super().__init__(model_state)
        self.update_dirs()

    def update_dirs(self):
        self.properties["SnapshotDirectory"] = self.properties["TrainingDirectory"] + "/snapshots/"
        self.properties["HyperParametersDirectory"] = self.properties["TrainingDirectory"] + "/hyperparam/"
        self.properties["HistoryDirectory"] = self.properties["TrainingDirectory"] + "/history/"

    def forward_prediction(self):
        """Do a forward prediction using the internal network and the internal image."""

        if PredictorLayer.show_images:
            pyplot.set_cmap(pyplot.gray())
#            pyplot.figure()
            pyplot.imshow(numpy.expand_dims(self.predictor_input_images[0], axis=3).squeeze())
            pyplot.title("input image")
            pyplot.show()
        y_pred_norm = self.model.predict(numpy.expand_dims(self.predictor_input_images[0], axis=3))
        return y_pred_norm

    def initialize_model(self):
        """Do one prediction step for the Network."""
        # Rotate the image, angle is the current prediction of the angle (as offset from 
        # inital values from JSON)

        if PredictorLayer.verbose:
            print("loading model:", self.properties["model_state"], "...")
        #input_shape = (48, 92, 1)
        #self.model = fitNet(input_shape)
        # self.model.load_weights(self.properties["KerasModelDirectory"])
        self.model = load_model(self.properties["model_state"])
        if PredictorLayer.verbose:
            print("keras model initialized sucessfully")
