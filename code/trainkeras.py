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

from keras_impl.parallel_model_save import ModelCheckpointParallel

from trainlayer import TrainLayer
from keraslayer import KerasLayer, KerasException
import h5py
import numpy
import json
import math

class TrainKeras(TrainLayer, KerasLayer):

	def __init__(self):
		TrainLayer.__init__(self)
		KerasLayer.__init__(self)


	def update_dirs(self):
		self.properties["SnapshotDirectory"] = self.properties["TrainingDirectory"] + "/snapshots/"
		self.properties["HyperParametersDirectory"] = self.properties["TrainingDirectory"] + "/hyperparam/"
		self.properties["HistoryDirectory"] = self.properties["TrainingDirectory"] + "/history/"
		
	def train(self, epochs, gpus=[0]):
		"""Train the network using tensorflow."""
	
		TrainLayer.train(self, epochs, gpus=gpus)

		filename = self.properties["Prefix"] + self.timestamp
		# safe the parameters
		self.writejson(self.properties["HyperParametersDirectory"] + filename + '.json')

		# load data
		training, test, image_shape, target_shape = self.load_data()
		# load model
		self.properties["InputShape"] = image_shape + (1,) # 1 channel
		self.properties["TargetShape"] = target_shape
		self.properties["Datasize"] = training[0].shape[0]
		self.load_model(gpus)
		# instantiate callbacks
		callbacks=[]
		if self.properties["SnapshotInterval"]:
			filepattern = self.properties["SnapshotDirectory"] + self.properties["Prefix"] + self.timestamp + "_{epoch:02d}.hd5"
			callbacks.append(ModelCheckpointParallel(
				filepath=filepattern,
				period=self.properties["SnapshotInterval"]
				))
			filepattern = self.properties["SnapshotDirectory"] + self.properties["Prefix"] + self.timestamp + "_best({epoch:02d}).hd5"
			cb = ModelCheckpointParallel(
				filepath=filepattern, 
				save_best_only=True
				)
			# skip writing snapshots for first epochs
			cb.epochs_since_last_save = 20 - epochs if epochs > 20 else 0
			callbacks.append(cb)
		

		# fixed or decay
		if self.properties["LRPolicy"] == "decay" or self.properties["LRPolicy"] == "fixed":
			pass # this is done by keras

		# step
		elif self.properties["LRPolicy"] == "step":
			#- step: return base_lr * gamma ^ (floor(iter / step))
			f = lambda epoch: self.properties["BaseLR"] * self.properties["Gamma"] ** (math.floor(epoch / self.properties["StepSize"]))
			callbacks.append(keras.callbacks.LearningRateScheduler(f))

		# auto
		elif self.properties["LRPolicy"] == "auto":
			raise KerasException('LRPolicy "auto" is not implemented')

		# exp
		elif self.properties["LRPolicy"] == "exp":
			#	- exp: return base_lr * gamma ^ iter
			f = lambda epoch: self.properties["BaseLR"] * self.properties["Gamma"] ^ epoch
			callbacks.append(keras.callbacks.LearningRateScheduler(f))

		# inv
		elif self.properties["LRPolicy"] == "inv":
			#- inv: return base_lr * (1 + gamma * iter) ^ (- power)
			f = lambda epoch: self.properties["BaseLR"] * (1. + self.properties["Gamma"] * epoch) ^ (-self.properties["Power"])
			callbacks.append(keras.callbacks.LearningRateScheduler(f))
#	- multistep: similar to step but it allows non uniform steps defined by
#	  stepvalue
#	- poly: the effective learning rate follows a polynomial decay, to be
#	  zero by the max_iter. return base_lr (1 - iter/max_iter) ^ (power)

		# sigmoid
		elif self.properties["LRPolicy"] == "sigmoid":
			#	- sigmoid: the effective learning rate follows a sigmod decay
			#	  return base_lr ( 1/(1 + exp(-gamma * (iter - stepsize))))
			f = lambda epoch: self.properties["BaseLR"] * (1./ (1. + math.exp(-self.properties["Gamma"] * (epoch - self.properties["StepSize"]))))
			callbacks.append(keras.callbacks.LearningRateScheduler(f))
		
		
		self.single_model.summary()

		self.history = self.model.fit(
			training[0], training[1], 
			validation_data=test, 
			epochs=epochs, 
			batch_size=self.properties["BatchSize"],
			callbacks=callbacks)

		# safe the history
		with open(self.properties["HistoryDirectory"] + filename + '_hist.json', "w") as file:
			json.dump(self.history.history, file, indent="\t")
		# safe the final model
		self.single_model.save(self.properties["SnapshotDirectory"] + filename + '.h5')

	def __prop__(self, key):
		return self.properties[key]
		
	def __set_prop__(self, key, item):
		self.properties[key] = item
		
	def load_data(self, image_key='images', target_key='targets'):
	
		images_train = list()
		images_test  = list()
		targets_train= list()
		targets_test = list()
	
		with open(self.properties["HDF5Directory"] + self.properties["H5TrainingListFile"], "r") as trainlist:
			for filename in trainlist:
				f = h5py.File(filename.strip(), "r")
				image_shape =  f[image_key].shape[2:]
				target_shape =  f[target_key].shape[1]
				images_train.append( f[image_key][()] )
				targets_train.append( f[target_key][()] )
				
		with open(self.properties["HDF5Directory"] + self.properties["H5TestListFile"], "r") as testlist:
			for filename in testlist:
				f = h5py.File(filename.strip(), "r")
				images_test.append( f[image_key][()] )
				targets_test.append( f[target_key][()] )
				
				
		#concat the data from multiple input files, if necessary
		if len(images_train) > 1:
			x_train = numpy.concatenate(images_train)
			y_train = numpy.concatenate(targets_train)
		else:
			x_train = images_train[0]
			y_train = targets_train[0]
			
		if len(images_test) > 1:
			x_test  = numpy.concatenate(images_test)
			y_test  = numpy.concatenate(targets_test)
		else:
			x_test = images_test[0]
			y_test = targets_test[0]

		### Input Shape
		#__Tensorflow has a different ordering than Caffe__

		#Tensorflow Ordering
		#``Rows x Cols x Channels``

		#Caffe Ordering
		#``Channels x Rows x Cols ``
		
	
		x_train = numpy.swapaxes(x_train,1,3)
		x_train = numpy.swapaxes(x_train,1,2)
		x_test = numpy.swapaxes(x_test,1,3)
		x_test = numpy.swapaxes(x_test,1,2)

		return (x_train, y_train), (x_test, y_test), image_shape, target_shape
