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

from matplotlib import pyplot
import numpy
import random
import math
import json
import operator
import time


class PredictorLayer:
	"""A abstract class to perform the prediction of points given a model."""
	
	show_images = False
	verbose     = False
	
	def __init__(self, model_state):
		"""Initialize the predictor layer with a solver and a model state.

		:param model_state: Path to the model / weights description of the network
		"""

		self.properties = { "model_state": model_state,
			"Iterations": 10 , "patterns" : { "tracked" : 'tracked{:08d}.JSON',
			"drr" : 'drr{:08d}.dcm', "parameter" : 'parameter{:08d}.JSON'},
			"Initial": {"Radius" : 2.5, "Angle" : 10.0},
			"TrainingDirectory" : ""}
		self.predictor_input_images = numpy.zeros((1, 1, 1, 1), dtype='float32')
		self.net_initialized = False
		self.image_loaded = -1
		
	def __setitem__(self, key, item):
		"""Sets a class property"""
		self.properties[key] = item
		self.update_dirs()
		
	def update_dirs(self):
		pass
		
	def __getitem__(self, key):
		"""Returns a class property"""
		return self.properties[key]

	def evaluate_one(self, imageindex, iterations):
		"""Evaluate the network iterations times to generate good estimations."""
		
		if self.image_loaded != imageindex:
			self.load_image(imageindex)
		return self._evaluate_one(iterations)
			
	def _evaluate_one(self, iterations):
		"""[Internal] evaluate current image."""
		self.generate_inital_state()
		
		data = {
			"input": {
				"x" : self.x, 
				"y" : self.y, 
				"angle" : self.angle, 
				"tilt": self.tilt, 
				"resolution": self.resolution },
			"ground_truth" : self.cut.get_ground_truth(),
			"intermediate" : []
			}
		for i in range(0, iterations):
			try:
				self.step()
				data["intermediate"].append({
					"x" : self.x, 
					"y" : self.y, 
					"angle" : self.angle, 
					"tilt": self.tilt, 
					"resolution": self.resolution,
					"p_old_style": self.p_old_style})
			except PredictorException as e:
				data["error"] = "Prediction failed on iteration " + str(i) + "\n" + e.message() 
				break
				
		data["output"] = data["intermediate"][-1]
		return data
		
	def _evaluate_one_repeat(self, iterations, repeats):
		"""[Internal] evaluate the image multiple times."""

		# TODO improve performance by running all repeats "at the same time"...
		
		data = []
		for i in range(0, repeats):
			data.append(self._evaluate_one(iterations))
		return data
	
	def evaluate_one_repeat(self, imageindex, iterations, repeats):
		"""Evaluate the network iteration times, repat repeat times with new setups."""
		if self.image_loaded != imageindex:
			self.load_image(imageindex)
		return self._evaluate_one_repeat(iterations, repeats)
	
	def evaluate(self, imageindices, iterations, repeats):
		data = []
		i = 0
		num = len(imageindices)
		
		print ("Evaluating " + str(num) + " images (" + str(repeats) + 
			" repeat(s), " + str(iterations) + " iteration(s)).")
		
		for index in imageindices:
			# projection = {}
			# if os.path.isfile(self.cut["parameters_file"]):
			try:
				data.append({"index": index, 
					"results": self.evaluate_one_repeat(index, iterations, repeats),
					"image": self.cut["dicom_file"],
					"projection": {}})
				with open(self.cut["ParameterDirectory"] + self.cut["parameters_file"], "r") as parameters_fp:
					data[-1]['projection'] = json.load(parameters_fp)
			except StopIteration:
				data=data[:-1]
			i = i + 1
			print ("Finished image " + str(i) + "/" + str(num))
		return data

	def set_state(self, angle, tilt, x, y, resolution = 0.0):
		"""Set the internal state of the current 2d pose, this is delta to internal cutter."""
		self.angle = angle
		self.tilt  = tilt
		self.x     = x
		self.y     = y
		self.resolution = resolution
		
	# def 
		
	def load_image(self, imageindex):
		"""Load a new image into memory."""
		#save image path and robot head position
		trackedJsonPath   = self.properties["patterns"]["tracked"].format(imageindex)
		parameterJsonPath = self.properties["patterns"]["parameter"].format(imageindex)
		imagePath         = self.properties["patterns"]["drr"].format(imageindex)
			
		# load json and image file
		self.cut.load(
			dicom_file   = imagePath, 
			tracked_file = trackedJsonPath,
			parameters_file = parameterJsonPath
			)
		self.image_loaded = imageindex
		
		if (self.predictor_input_images.shape[2] != self.cut["Final"]["Height"] 
			or self.predictor_input_images.shape[3] != self.cut["Final"]["Width"]):
			self.predictor_input_images = (
				numpy.zeros((1, 1, 
					self.cut["Final"]["Height"], 
					self.cut["Final"]["Width"]), 
					
					dtype='float32'))
		
		self.norms = self.cut.calculate_norms()

#		self.estimate

	def generate_inital_state(self, labelAccuracy=None, labelAngleAccuracy=None):
		"""Randomly generate a new state."""
		
		if labelAccuracy == None:
			labelAccuracy = self.properties["Initial"]["Radius"]
		
		if labelAngleAccuracy == None:
			labelAngleAccuracy = self.properties["Initial"]["Angle"]
		
		if labelAccuracy > 0.0:

			# randomRadius = random.uniform(0, labelAccuracy) # in pixel
			randomRadius = random.uniform(0, labelAccuracy / self.cut.get_ground_truth()["resolution"]) # in mm
			randomPhi    = random.uniform(0,360) # in degrees

			self.x = self.cut.get_ground_truth()["x"] + randomRadius * math.cos(math.radians(randomPhi))
			self.y = self.cut.get_ground_truth()["y"] + randomRadius * math.sin(math.radians(randomPhi))
			
			if PredictorLayer.verbose:
				print ("generated start point: (" + str(self.x) + " px, " + str(self.y) + " px)")
				print ("deviation : (" + str(self.x - self.cut.get_ground_truth()["x"]) + " px, " + str(self.y - self.cut.get_ground_truth()["y"]) + " px)")
		else:
			self.x = self.cut.get_ground_truth()["x"]
			self.y = self.cut.get_ground_truth()["y"]
			
		# new angle for calculating the points for the orientation
		if labelAngleAccuracy > 0.0:
			self.angle = (min(3*labelAngleAccuracy, max(-3*labelAngleAccuracy, 
				numpy.random.normal(0.0, labelAngleAccuracy))) 
				
				+ self.cut.get_ground_truth()["angle"])
				
			if PredictorLayer.verbose:
				print ("generated angle noise: " + str(self.angle) + " deg")
				print ("deviation: " + str(self.angle - self.cut.get_ground_truth()["angle"]) + " deg")
		else:
			self.angle = self.cut.get_ground_truth()["angle"]
		
		# since we do not use tilt for initial values, 0 is fine
		self.tilt = 0.0
		self.resolution = (self.cut["MinResolution"] + self.cut["MaxResolution"] ) / 2
		
	def forward_prediction(self):
		"""Do a forward prediction using the internal network and the internal image."""
		raise PredictorException("The subclass of PredictorLayer should implement forward_prediction.")

	def initialize_model(self):
		"""Initialize a network after loading a new file."""
		raise PredictorException("The subclass of PredictorLayer should implement initialize_model.")

	def step(self):
		"""Do one prediction step for the Network."""
		# Rotate the image, angle is the current prediction of the angle (as offset from 
		# inital values from JSON)

		if not self.net_initialized:
			self.initialize_model()
			self.net_initialized = True

		# propagate current values to the cutter
		self.cut.set_state(
			x = self.x, 
			y = self.y, 
			angle = self.angle, 
			tilt = self.tilt, 
			resolution = self.resolution)
		
		self.cut.cut_to_large()
		self.cut.rotate(0.0)
			
		# cut to the current / inital prediction (again as difference from initial values fom JSON)
		image = self.cut.cut_to( (0.0, 0.0) )
		
		# the search diverged
		if image.shape[0] != 48 or image.shape[1] != 92:
			message = ("The prediction most likely diverged as it cannot find the correct " +
				"image patch any more.\n\tshape: " + str(image.shape) + "\n"
				'The prediction failed on image ' + str(self.cut.dicom_file))
			print(message)
			raise PredictorException(message)
			
		
		self.predictor_input_images[0] = image
		#target = [newX, newY, xPoint1, yPoint1, xPoint2, yPoint2, xPoint3, yPoint3]
		y_pred_normal = self.forward_prediction()
		
		if PredictorLayer.verbose or PredictorLayer.show_images:
			targets = self.cut.targets()
			gt_targets = self.cut.get_ground_truth_targets()
			
		if PredictorLayer.verbose:
			print ("predicted normalized positions:")
			print (y_pred_normal)
			
			print ("ground truth normalized positions:")
			print (self.cut.normalize(gt_targets))
		
		if self.cut["Mode"] == "end-to-end":
			if PredictorLayer.verbose or PredictorLayer.show_images:
				targets = targets[0:-1]
				gt_targets = gt_targets[0:-1]
			
			
			y_pred =  self.cut.unnormalize( 
				tuple( 
					(y_pred_normal[0][i], y_pred_normal[0][i+1]) for i in range(0, len(y_pred_normal[0])-3, 2)
					) + (y_pred_normal[0][len(y_pred_normal[0])-3:],)
				)
			
			y_pred_full = self.cut.transform_final_to_full( y_pred[0] )
			
			angleDif = y_pred[-1][0]
			tilt = y_pred[-1][1]
			res = y_pred[-1][2]
		else:
			y_pred =  self.cut.unnormalize( 
				tuple( 
					(y_pred_normal[0][i], y_pred_normal[0][i+1]) for i in range(0, len(y_pred_normal[0]), 2) 
					) 
				)

			y_pred_full, angleDif, tilt, res = self.points_to_state(y_pred)

		if PredictorLayer.verbose:
			print ("predicted pixel-positions:")
			print (y_pred)
			
			print ("ground truth pixel-positions:")
			print (gt_targets)
			
			print ("The calculated angle error is " + str(angleDif) + " deg")
			
		if PredictorLayer.show_images:
			pyplot.set_cmap(pyplot.gray())
			pyplot.imshow(1-image)
			pyplot.axis("off")
			#pyplot.title("Input image, ground truth position in green, input condition in blue, prediction in orange")
			
			#pyplot.scatter(tuple(gt[0] for gt in gt_targets), tuple(gt[1] for gt in gt_targets), marker='x', s=50, color='green')
			#pyplot.scatter(tuple(t[0] for t in targets), tuple(t[1] for t in targets), marker='x', s=50, color='blue')
			pyplot.scatter(tuple(x[0] for x in y_pred), tuple(x[1] for x in y_pred), marker='x', s=60, color='orange')
		
		if PredictorLayer.verbose:
			difference = ( y_pred_full[0] - self.x, y_pred_full[1] - self.y )
			print ("predicted offset is: " + str(difference) + " px")
			
		# update scheme change the way the values are updated.
		if res != None:
			self.update(y_pred_full, self.angle - angleDif, tilt, res, self.cut.transform_final_to_full(y_pred[0]) )
			if PredictorLayer.show_images:
				p = self.cut.transform_full_to_final( (self.x, self.y) )
				#pyplot.scatter(p[0], p[1], marker='x', s=50, color='red')
		else:
			self.update(y_pred_full, self.angle - angleDif, tilt)

		if PredictorLayer.verbose:
			print ("predicted value is: " + str(self.x) + " " + str(self.y) )

			gt = self.cut.get_ground_truth()
			
			if res == None:
				real_tilt_acos = tilt_approx * gt["resolution"]
				if abs(real_tilt_acos) > 1.1:
					print ("WARNING: values very inconsistent for tilt calculation. acos > 1.1")
				if abs(real_tilt_acos) > 1.0:
					real_tilt = 0.
				else:
					real_tilt = math.degrees(math.acos( real_tilt_acos ) )
				print ("predicted tilt: " + str(real_tilt))
			
			print ("Error: ", self.x - gt["x"], " ", self.y - gt["y"], 
				", ", self.angle - gt["angle"], " tilt: ", (real_tilt if res == None else tilt) - abs(gt["tilt"]))
				
		if PredictorLayer.show_images:
			pyplot.savefig("viz_images/viz_"+str(time.time())+".png")
			pyplot.show()

	def points_to_state(self, y_pred):
		"""Calculate state variables from TargetDistances-Points."""
			
		# line of best fit
		y_pred_on_x = []
		tilt_approx = []
		for i, item in enumerate(y_pred):
			if self.cut["TargetDistances"][i][1] == 0.:
				y_pred_on_x.append(item)
				if self.cut["TargetDistances"][i][0] != 0.:
					tilt_approx.append( math.sqrt(
						( (item[0] - y_pred[0][0]) / self.cut["TargetDistances"][i][0]) **2 +
						( (item[1] - y_pred[0][1]) / self.cut["TargetDistances"][i][0]) **2 ) )
		if len(tilt_approx) > 1:
			tilt_approx = numpy.average(tilt_approx)
		else:
			tilt_approx = tilt_approx[0]
		
		x = tuple(p[0] for p in y_pred_on_x)
		y = tuple(p[1] for p in y_pred_on_x)
		
		m, b = numpy.polyfit(x, y, 1)
		
		# use a fit in x- and y- points
		if len(y_pred_on_x) != len(y_pred):
			# line of best fit to position center at 
			y_pred_on_y = []
			res = []
			for i, item in enumerate(y_pred):
				if self.cut["TargetDistances"][i][0] == 0.:
					y_pred_on_y.append(item)
					if self.cut["TargetDistances"][i][1] != 0.:
						res.append( 1 / math.sqrt(
							( (item[0] - y_pred[0][0]) / self.cut["TargetDistances"][i][1]) **2 +
							( (item[1] - y_pred[0][1]) / self.cut["TargetDistances"][i][1]) **2 ) )
			if len(res) > 1:
				res = numpy.average(res)
			else:
				res = res[0]
			
			x1 = tuple(p[0] for p in y_pred_on_y)
			y1 = tuple(p[1] for p in y_pred_on_y)
			
			m1, b1 = numpy.polyfit(x1, y1, 1)
			t = -(b1 - b) / (m1 - m)
			y_pred_full = self.cut.transform_final_to_full( (t, m*t + b) )
		else:
			# transform the predictions back to "full" coordinates
			y_pred_full = self.cut.transform_final_to_full( y_pred[0] )
			
			tilt = None
			res = None
		
		# Approximate the angle from the slope of the fitted line
		angleDif = -math.degrees(math.atan(m))
		# Approximate the tilt from the distances of the fitted line
		if res != None:
			tilt_acos = tilt_approx * res
		else:
			tilt_acos = tilt_approx * self.cut["MinResolution"]
		
		if abs(tilt_acos) > 1.1:
			print ("WARNING: values very inconsistent for tilt calculation. acos > 1.1")
		if abs(tilt_acos) > 1.0:
			tilt = 0.
		else:
			tilt = math.degrees(math.acos( tilt_acos ))

		return y_pred_full, angleDif, tilt, res
		
	def update(self, p, angle, tilt, resolution = None, p_old_style = None):
		"""update the internal state for next iteration."""
		
		## if we want to use a stepsize, it should be in here

		# update current head angle and position
		# currentAngle = currentAngle + math.exp(-0.25 * k) * angleDif
		# currentX = currentX + math.exp(-0.25 * k) * difference[0]
		# currentY = currentY + math.exp(-0.25 * k) * difference[1]
		
		self.x = p[0]
		self.y = p[1]
		self.angle = angle
		self.tilt = tilt
		if resolution != None:
			self.resolution = resolution
		self.p_old_style = p_old_style
	
	def set_cutter(self, cutter):
		self.cut = cutter
		
class PredictorException (BaseException):
	def __init__(self, message):
		self.s_message = message
	
	def message(self):
		return self.s_message
