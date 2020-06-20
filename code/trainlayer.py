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

from datetime import datetime
import json

class TrainLayer:

	def __init__(self):
		self.properties = { "LRPolicy" : "fixed", "TrainingDirectory" : "", "HDF5Directory": "",
			"Prefix": "", "WeightDecay": 1e-4, "H5TrainingListFile": "", "H5TestListFile": "", 
			"BatchSize": 512, "Prefix": "UnnamedNetwork", "DataSize": 0, "TimestampFormat": '%Y%m%d_%H%M',
			"Gamma": 1.0, "Power": 1.0, "StepSize": 5, "BaseLR": 1e-2, "SnapshotInterval": 5,
			"Resume": "", "Momentum": 0.9
			}
		
	def update_dirs(self):
		"""Update the dir properties of the values."""
		pass
		
	def train(self, epochs, gpus=1):
		"""Train the network with the GPUs in gpuOption."""

		self.timestamp = datetime.today().strftime(self.properties["TimestampFormat"])
		
	def __setitem__(self, key, item):
		"""Sets a class property"""
		
		self.properties[key] = item
		
		if key == "TrainingDirectory":
			self.update_dirs()
		
	def __getitem__(self, key):
		"""Returns a class property"""
		return self.properties[key]

	def writejson(self, filename):
		with open(filename, "w") as file:
			json.dump(self.properties, file, indent="\t")