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

import keras
from keras.models import Sequential
from keras.layers import *

def i3PosNet(input_shape, out_number=8):

	model = Sequential()
	# Input Layer and 5 Conv Layers
	model.add(Conv2D(20, (3, 3), activation='relu', input_shape=input_shape, name="conv2d_1"))
	model.add(Conv2D(32, (9, 9), activation='relu', name="conv2d_2"))
	model.add(Conv2D(32, (9, 9), activation='relu', name="conv2d_3"))
	model.add(Conv2D(32, (9, 9), activation='relu', name="conv2d_4"))
	model.add(Conv2D(32, (9, 9), activation='relu', name="conv2d_5"))
	# Fully Connected Part
	model.add(Flatten(name="flatten_1"))
	model.add(Dense(160, bias_initializer=keras.initializers.Constant(value=0), name="dense_1" ))
	model.add(Dense(out_number, bias_initializer=keras.initializers.Constant(value=0), name="dense_2"  ))
	# return the created instance
	return model

def i3PosNetVGG(input_shape, out_number=8, layer_count=(2,2), fc_layer_count = (1,4), fc_reg='None', conv_reg='None', shrinking='MaxPooling', padding="same"):
	"""Create a parametrizable i3PosNet network that is VGG-like.
	
	
	fc_layer_count : (<number of fully connected layers>, <factor for conversion after flattening>)
	"""

	model = Sequential()
	feature_count = 32
	model.add(Conv2D(feature_count, input_shape=input_shape, kernel_size=(3, 3), activation='relu', padding=padding))
	# Blocks
	for l in layer_count:
		if shrinking == "Stride":
			l -= 1
		for i in range(0,l):
			model.add(Conv2D(feature_count, kernel_size=(3, 3), activation='relu', padding=padding))
			if conv_reg == "BatchNormalization_Layer":
				model.add(BatchNormalization(axis=-1, momentum=0.99))
		if shrinking == "MaxPooling":
			model.add(MaxPooling2D(pool_size=(2,2)))
		elif shrinking == "AveragePooling":
			model.add(AveragePooling2D(pool_size=(2,2)))
		elif shrinking == "Stride":
			model.add(Conv2D(feature_count, kernel_size=(3, 3), activation='relu', strides=(2,2), padding=padding))
		feature_count *= 2
		
		if conv_reg == "Dropout":
			model.add(Dropout(0.2))
		elif conv_reg == "BatchNormalization_Block":
			model.add(BatchNormalization(axis=-1, momentum=0.97))
	
	feature_count *= int((fc_layer_count[1] / 2))
	
	# Fully Connected Layers
	model.add(Flatten())
	if fc_reg == "Dropout":
		model.add(Dropout(0.2))
	for i in range(0, fc_layer_count[0]):
		model.add(Dense(feature_count, activation="relu"))
		if fc_reg == "Dropout":
			model.add(Dropout(0.2))
		elif fc_reg == "BatchNormalization":
			model.add(BatchNormalization(momentum=0.99))
		elif fc_reg == "BatchNormalization Dropout":
			model.add(BatchNormalization(momentum=0.99))
			model.add(Dropout(0.1))
		elif fc_reg == "BatchNormalization Dropout 0.05":
			model.add(BatchNormalization(momentum=0.99))
			model.add(Dropout(0.05))
		elif fc_reg == "BatchNormalization Dropout 0.2":
			model.add(BatchNormalization(momentum=0.99))
			model.add(Dropout(0.2))
		feature_count = int(feature_count/2)
	
	model.add(Dense(out_number, bias_initializer=keras.initializers.Constant(value=0)))
	return model
