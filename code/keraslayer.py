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

import tensorflow
from collections import Sequence
from keras_impl.model_definition import *
from keras.utils import multi_gpu_model
from keras.optimizers import SGD, Adam

class KerasLayer():
	def __init__(self):
		"""Initialize the model."""
		self.model = None
		self.single_model = None
		self.__set_prop__("InputShape", (96,48))
		self.__set_prop__("TargetShape", 8)
		self.__set_prop__("LayerCount", (2,2))
		self.__set_prop__("FCLayerCount", 2)
		self.__set_prop__("FCReg", "None")
		self.__set_prop__("ConvReg", "None")
		self.__set_prop__("Shrinking", "None")
		self.__set_prop__("Optimizer", "Adam")
		self.__set_prop__("Padding", "valid")
		self.__set_prop__("Beta1", 0.9)
		self.__set_prop__("Beta2", 0.999)
		self.__set_prop__("Epsilon", None)
	
	def __prop__(self, key):
		"""Return a property."""
		return None
		
	def __set_prop__(self, key, item):
		"""Set a property."""
		pass
	
	def set_model(self, model):
		"""Set the model."""
		if model == None:
			if self.model != None:
				del self.model
				self.model = None
			if self.single_model != None:
				del self.single_model
				self.single_model = None
		else:
			self.model = model
			del self.single_model
			self.single_model = None
	
	def get_model(self):
		return self.model

	def load_model(self, gpus=1):
		"""Load a model, if none is set. Either use Resume property file or generate."""
		
		if self.model != None:
			return

		## build the model on the CPU if parallelism is targeted
		if isinstance(gpus, Sequence):
			if len(gpus) != 1:
				device = "/cpu:0"
				multigpu = True
			else:
				device = "/gpu:{:d}".format(gpus[0])
				multigpu = False
		else:
			if gpus != 1:
				device = "/cpu:0"
				multigpu = True
			else:
				device = "/gpu:{:d}".format(gpus)
				multigpu = False


		if self.__prop__("Resume"):
			self.model = keras.models.load_model(
				self.__prop__("SnapshotDirectory") + self.__prop__("Prefix") + self.__prop__("Resume") + '.h5"')
			self.single_model = self.model
			if multigpu:
				self.model = multi_gpu_model(self.model, gpus)
		else: 
			
			with tensorflow.device(device):
				if self.__prop__("Prefix").startswith("i3PosNet_VGG16"):
					self.model = i3PosNetVGG(
						input_shape=self.__prop__("InputShape"), 
						out_number=self.__prop__("TargetShape"),
						layer_count=self.__prop__("LayerCount"), 
						fc_layer_count=self.__prop__("FCLayerCount"), 
						fc_reg=self.__prop__("FCReg"), 
						conv_reg=self.__prop__("ConvReg"), 
						shrinking=self.__prop__("Shrinking"),
						padding=self.__prop__("Padding"))
				else:
					self.model = i3PosNet(image_shape, out = self.__prop__("TargetShape"))

			self.single_model = self.model
			if multigpu:
				self.model = multi_gpu_model(self.model, gpus)
				
			# clear model
			# try:
				# del self.model
			# except:
				# pass

			if self.__prop__("Optimizer") == "SGD":
				optimizer = SGD(
					lr=self.__prop__("BaseLR"),
					decay=self.__prop__("Gamma") if self.__prop__("LRPolicy") == "decay" else 0.,
					momentum= self.__prop__("Momentum"),
					nesterov=True)
			elif self.__prop__("Optimizer") == "Adam":
				optimizer = Adam(
					lr=self.__prop__("BaseLR"),
					decay=self.__prop__("Gamma") if self.__prop__("LRPolicy") == "decay" else 0.,
					# use defaults for these for now (b1 = 0.9, b2 = 0.999, e = 1e-8
					beta_1=self.__prop__("Beta1"),
					beta_2=self.__prop__("Beta2"),
					epsilon=self.__prop__("Epsilon")
					)
			
			self.model.compile(loss='mean_squared_error', optimizer=optimizer)


class KerasException(BaseException):
	def __init__(self, message):
		self.s_message = message
	
	def message(self):
		return self.s_message
