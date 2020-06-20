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

ort dicom
from matplotlib import pyplot
import numpy
import json
import math
from scipy import ndimage

class Cutter:
    """A class to cut an image to a specific desgin."""

    show_images = False
    verbose     = False

    def __init__(self):
        self.properties = { "Large": {"Height": 180, "Width": 180}, 
            "CenterOffset": (-15,0), "Final": {"Height": 48, "Width": 92},
            "MaxDisplacement": 2.5, "MaxAngleRotation": 30, "MinResolution": 0.1, 
            "ImageDirectory": "", "TrackedDirectory": "", "OutputDirectory": "",
            "ParameterDirectory": "", "Mode": "modular", "MaxResolution": 0.2, 
            "TargetDistances": [(0.0,0.0),(1.0,0.0),(-1.0,0.0)],
            "InvertIntensities": False}
        self.norms_modified = True

    def load(self, dicom_file, tracked_file, parameters_file=""):
        self.properties['dicom_file'] = dicom_file
        self.properties['tracked_file'] = tracked_file
        self.properties['parameters_file'] = parameters_file
        # load file data
        self.load_json(tracked_file)
        self.load_dicom(dicom_file)

        if self.verbose:
            print ('current orientation of instrument: ' + str(self.angle))

        self.cut_to_large()

    def get_resolution(self):
        return self.resolution

    def get_angle(self):
        """Returns the angle the object is turned."""
        return self.angle

    def get_large(self):
        return (self.large_x, self.large_y)

    def __setitem__(self, key, item):
        """Sets a class property"""

        if key == "TargetDistances":
            # if this uses the "old syntax of positions on the x-axis, try to convert
            if not hasattr(item[0], '__iter__'):
                item = list((0., i) for i in item)
            if item[0] != (0.0,0.0):
                raise CutterException("The first item of 'TargetDistances' has to be (0.0,0.0)!")

        self.norms_modified = True
        self.properties[key] = item

    def __getitem__(self, key):
        """Returns a class property"""
        return self.properties[key]

    def target_number(self):
        """Returns the number of values per object."""
        return int(len(self.properties["TargetDistances"]) * 2 + (3 if self.properties["Mode"] == "end-to-end" else 0))

    def load_dicom(self, filename):
        """Loads a dicom file into memory"""
        # load dicom image
        self.image = dicom.read_file(self.properties["ImageDirectory"] + filename)
        self.dicom_file = filename

    def load_json(self, filename):
        """Loads a json file containing tracked Point information into memory."""
        # load json file
        with open(self.properties["TrackedDirectory"] + filename) as tracked_file:
            json_data = json.load(tracked_file)
            self.full_x= json_data['Image']['HeadX']
            self.full_y= json_data['Image']['HeadY']
            self.angle = json_data['Image']['HeadAngle']
            self.angleRad = math.radians(self.angle)
            self.tilt  = json_data['Image']['HeadTilt']
            self.resolution = json_data['Image']['ResolutionAtHead']
        self.values_are_ground_truth = True
        self.ground_truth = { "x" : self.full_x, "y" : self.full_y, "angle": self.angle, 
            "tilt" : self.tilt, "resolution" : self.resolution}

    def set_state(self, x, y, angle, tilt, resolution):
        """Set the internal state of the cutter."""
        self.full_x = x
        self.full_y = y
        self.angle = angle
        self.angleRad = math.radians(self.angle)
        self.tilt  = tilt
        self.resolution = resolution
        self.values_are_ground_truth = False

    def get_state(self):
        """Return the internal state of the cutter."""
        return {
            "x" : 		self.full_x, 
            "y" : 		self.full_y, 
            "angle" : 	self.angle, 
            "tilt": 	self.tilt, 
            "resolution": self.resolution }

    def get_ground_truth(self):
        """Return the internal ground truth state of the cutter."""

        return self.ground_truth

    def get_error(self):
        """Return the difference between the current internal state and the ground truth."""

        return {
            "x" : 		self.full_x - self.ground_truth.x, 
            "y" : 		self.full_y - self.ground_truth.y, 
            "angle" : 	self.angle  - self.ground_truth.angle, # positive is counter-clockwise deviation
            "tilt": 	self.tilt   - self.ground_truth.tilt, 
            "resolution": self.resolution  - self.ground_truth.resolution}

    def set_state_ground_truth(self):
        """Set the internal values of the cutter to ground truth."""
        self.set_state(
            x =     self.ground_truth.x, 
            y =     self.ground_truth.y, 
            angle = self.ground_truth.angle, 
            tilt =  self.ground_truth.tilt, 
            resolution=self.ground_truth.resolution)
        self.values_are_ground_truth = True

    def cut_to_large(self):
        """Takes the currently loaded image and cuts it to "Large" Dimensions."""
        height_large = int(self.properties["Large"]["Height"])
        width_large  = int(self.properties["Large"]["Width"])

        # show image (if selected)
        if self.show_images:
            pyplot.set_cmap(pyplot.gray())
            pyplot.imshow(self.image.pixel_array)
            pyplot.title("Full image with large cutout in blue box, \n" + 
                ("position of center in green" if self.values_are_ground_truth else 
                "assumed position of center in blue"))
            # position marker in blue
            pyplot.scatter(self.full_x, self.full_y, marker='x', s=50, color='green' if self.values_are_ground_truth else 'blue')
            # borders in blue
            pyplot.plot(
                (self.full_x-height_large/2, self.full_x-height_large/2, self.full_x+height_large/2, self.full_x+height_large/2, self.full_x-height_large/2), 
                (self.full_y-width_large/2,  self.full_y+width_large/2,  self.full_y+width_large/2,  self.full_y-width_large/2,  self.full_y-width_large/2), 
                linestyle='solid',linewidth=1, color='blue')
            pyplot.show()

        # crop image for rotation
        self.large_top  = int(round(self.full_y - height_large/2))
        bottom     		= self.large_top + height_large
        self.large_left = int(round(self.full_x - width_large/2))
        right           = self.large_left + width_large
        
        # raise an Exception, if out of view
        if (self.large_top < 0) or (self.large_left < 0) or (bottom > self.image.pixel_array.shape[0]-1) or (right > self.image.pixel_array.shape[1]-1):
            raise StopIteration()

        self.image_large = self.image.pixel_array[self.large_top:bottom, self.large_left:right]
        
        if self.properties["InvertIntensities"]:
            self.image_large = -1*self.image_large

        # calculate new head point
        self.large_x, self.large_y = self.transform_full_to_large((self.full_x, self.full_y))

        if self.show_images:
            pyplot.set_cmap(pyplot.gray())
            pyplot.imshow(self.image_large)
            pyplot.title("Large image with " + 
                ("position of center in green" if self.values_are_ground_truth else 
                "assumed position of center in blue"))
            # position marker in blue
            pyplot.scatter(self.large_x, self.large_y, marker='x', s=50, color='green' if self.values_are_ground_truth else 'blue')
            pyplot.show()

    #finds the straight-line distance between two points
    @staticmethod
    def distance(ax, ay, bx, by):
        return math.sqrt((by - ay)**2 + (bx - ax)**2)

    #rotates point `A` about point `B` by `angle` radians clockwise.
    @staticmethod
    def rotated_about(ax, ay, bx, by, angle):
        radius = Cutter.distance(ax,ay,bx,by)
        angle += math.atan2(ay-by, ax-bx)
        return (
            bx + radius * math.cos(angle),
            by + radius * math.sin(angle)
        )

    def transform_full_to_large(self, p):
        """Transform point p as coordinates in full to coordinates in large."""
        return (p[0] - self.large_left, p[1] - self.large_top)

    def transform_large_to_full(self, p):
        """Transform point p as coordinates in large to coordinates in full."""
        return (p[0] + self.large_left, p[1] + self.large_top)

    def transform_large_to_rotated(self, p):
        """Transform point p as coordinate in large to coordinate in rotated."""
        height_large = int(self.properties["Large"]["Height"])
        width_large  = int(self.properties["Large"]["Width"])

        # rotated p about center of image by -angle (since we are correcting for the rotation 
        # done clockwise by ndimage - so correction should be counter-clockwise)
        return Cutter.rotated_about(p[0], p[1], width_large/2 - 0.5, height_large/2 - 0.5, -self.rotated_angle_rad)

    def transform_rotated_to_large(self, p):
        """Transform point p as coordinate in rotated to coordinate in large."""
        height_large = int(self.properties["Large"]["Height"])
        width_large  = int(self.properties["Large"]["Width"])

        # rotated p about center of image by -angle (since we are correcting for the rotation 
        # done clockwise by ndimage - so correction should be counter-clockwise)
        return Cutter.rotated_about(p[0], p[1], width_large/2 - 0.5, height_large/2 - 0.5, self.rotated_angle_rad)

    def transform_rotated_to_final(self, p):
        """Transform point p as coordinate in rotated to coordinate in final."""
        return (p[0] - self.get_final_left(), p[1] - self.get_final_top())

    def transform_final_to_rotated(self, p):
        """Transform point p as coordinate in final to coordinate in rotated."""
        return (p[0] + self.get_final_left(), p[1] + self.get_final_top())

    def transform_full_to_final(self, p):
        """Transform point p as coordinate in full to coordinate in final."""
        return self.transform_rotated_to_final(self.transform_large_to_rotated(self.transform_full_to_large(p)))

    def transform_final_to_full(self, p):
        """Transform point p as coordinate in final to coordinate in full."""
        return self.transform_large_to_full(self.transform_rotated_to_large(self.transform_final_to_rotated(p)))

    def rotate(self, angle, order=3):
        """Takes the current "Large" image and rotates it by angle + ground_truth angle from json resulting 
        in a "Rotated" image, where the object is rotated by angle degrees against the x-axis."""
        self.image_rotated = ndimage.rotate(self.image_large, angle + self.angle, reshape=False, order=order) # angle in degrees

        if self.verbose:
            print ('reorient to angle ' + str(angle) + ' deg')

        # rotated Point still holds the ground truth information, that is ground truth for image_rotated
        self.rotated_angle_rad = math.radians(angle + self.angle)
        self.rotated_x, self.rotated_y = self.transform_large_to_rotated(
            (self.large_x, self.large_y) )

        if self.verbose:
            print ('pre rotation:  ' + str((self.large_x, self.large_y)))
            print ('post rotation: ' + str((self.rotated_x, self.rotated_y)))

        self.final_angle = angle

    def get_final_left(self):
        return self.final_left

    def get_final_top(self):
        return self.final_top

    def cut_to(self, displacement):
        """Cuts the image to the final designated "design". 

        image = cutter.cut_to(displacement) , 

        where 
        displacement 	is the distance the object can have from the "optimal" position as a vector.
        """

        pixel_displacement = (displacement[0] / self.resolution, displacement[1] / self.resolution)

        height_final = int(self.properties["Final"]["Height"])
        width_final  = int(self.properties["Final"]["Width"])
        center_offset= self.properties["CenterOffset"]

        self.final_left   = int(round(self.rotated_x - center_offset[0] - width_final/2  + 0.5 + pixel_displacement[0]))
        self.final_top    = int(round(self.rotated_y - center_offset[1] - height_final/2 + 0.5 + pixel_displacement[1]))
        final_right  = int(self.final_left + width_final)
        final_bottom = int(self.final_top + height_final)

        if self.show_images:
            draw_circle = any(abs(d) > 1e-8 for d in displacement)
            pyplot.set_cmap(pyplot.gray())
            pyplot.imshow(self.image_rotated)
            pyplot.title("Rotated image with final cutout in blue box, " + 
                ("position of center in green" if self.values_are_ground_truth else 
                "assumed position of center in blue") + 
                (", \nscattering circle (w.r.t. cutout) in red plus circle center as red dot" if draw_circle else ""))
            # position marker of the center in blue
            pyplot.scatter(self.rotated_x, self.rotated_y, marker='x', s=50, color='green' if self.values_are_ground_truth else 'blue')
            # border around the 90x46 image in red
            # define corner points:
            pyplot.plot(
                (self.final_left, self.final_left, final_right, final_right, self.final_left),
                (self.final_top, final_bottom, final_bottom, self.final_top, self.final_top), 
                linestyle='solid',linewidth=1, color='blue')
            if draw_circle:
                pyplot.scatter(self.final_left + width_final/2 + center_offset[0], self.final_top + height_final/2 + center_offset[1], marker='o', s=15, color='red')
                circle1 = pyplot.Circle(
                    (self.final_left + width_final/2 + center_offset[0], self.final_top + height_final/2 + center_offset[1]), 
                    self.properties["MaxDisplacement"] / self.resolution, color='red', fill=False)
#			circle2 = pyplot.Circle((self.rotated_x, self.rotated_y, pixel_displacement, color='blue', fill=False)
            fig = pyplot.gcf()
            ax = fig.gca()
#			ax.add_artist(circle2)
            if draw_circle:
                ax.add_artist(circle1)
            pyplot.show()

        # crop image to heightFinal x widthFinal
        self.image_final = self.image_rotated[self.final_top:final_bottom, self.final_left:final_right]

        if self.verbose:
            print ('cut image to dimensions: ' + str(self.image_final.shape))

        # calculate new head point in the cropped image
        self.final_x, self.final_y = self.transform_rotated_to_final((self.rotated_x, self.rotated_y))

        if self.show_images:
            pyplot.set_cmap(pyplot.gray())
            pyplot.imshow(self.image_final)
            pyplot.title("Final image with " + 
                ("position of center in green" if self.values_are_ground_truth else 
                "assumed position of center in blue"))
            pyplot.scatter(self.final_x, self.final_y, marker='x', s=50, color='green' if self.values_are_ground_truth else 'blue')
            pyplot.show()

        # normalize and save the cropped image in the images array
        self.image_normalized = self.image_final.astype('float32')
        image_min = self.image_normalized.min()
        image_max = self.image_normalized.max()

        return numpy.divide(self.image_normalized - image_min, image_max - image_min)

    def targets(self):
        """Uses the last images settings to calculated the corresponding intended target values.

        targets = cutter.targets([positionError], [angleError], [tiltError])

        where
        targets 	is a list of 2d tuples of the x and y coordinates in pixel values."""


        if self.verbose:
            print ("Tilt: " + str(self.tilt) + " Angle: " + str(self.final_angle) + " Resolution: " + str(self.resolution))

        points = []
        costilt = math.cos(math.radians(self.tilt))
        sinangle = math.sin(math.radians(self.final_angle))
        cosangle = math.cos(math.radians(self.final_angle))
        for i, dist in enumerate(self.properties["TargetDistances"]):
            # iterate to create the Points for the targets
            points.append( 
                ( self.final_x + (dist[0] * cosangle * costilt + dist[1] * sinangle) / self.resolution,
                  self.final_y + (dist[1] * cosangle - dist[0] * sinangle * costilt) / self.resolution ) )

        if self.show_images:
            pyplot.set_cmap(pyplot.gray())
            pyplot.imshow(self.image_final)
            pyplot.title("Final image with " + 
                ("target points in green" if self.values_are_ground_truth else 
                "assumed target points in blue"))
            pyplot.scatter(
                tuple(p[0] for p in points), 
                tuple(p[1] for p in points), 
                marker='x', s=50, 
                color='green' if self.values_are_ground_truth else 'blue')
            pyplot.show()

        if self.properties["Mode"] == "end-to-end":
            points.append((self.final_angle, self.tilt, self.resolution))
        return points

    def get_ground_truth_targets(self):
        """Uses the last images settings to calculated the corresponding intended target values.

        targets = cutter.targets([positionError], [angleError], [tiltError])

        where
        targets 	is a list of 2d tuples of the x and y coordinates in final-pixel values."""


        gt_x, gt_y = self.transform_full_to_final((self.ground_truth["x"], self.ground_truth["y"]))
        gt_angle   = (self.final_angle + self.angle) - self.ground_truth["angle"]
        gt_tilt    = self.ground_truth["tilt"]
        gt_resolution = self.ground_truth["resolution"]

        points = []
        gt_costilt = math.cos(math.radians(gt_tilt))
        gt_sinangle = math.sin(math.radians(gt_angle))
        gt_cosangle = math.cos(math.radians(gt_angle))
        for i, dist in enumerate(self.properties["TargetDistances"]):
            # iterate to create the Points for the targets
            points.append( 
                ( gt_x + (dist[0] * gt_cosangle * gt_costilt + dist[1] * gt_sinangle) / gt_resolution,
                  gt_y + (dist[1] * gt_cosangle - dist[0] * gt_sinangle * gt_costilt) / gt_resolution ) )

        if self.show_images:
            pyplot.set_cmap(pyplot.gray())
            pyplot.imshow(self.image_final)
            pyplot.title("Final image with ground truth target points in green")
            pyplot.scatter(tuple(p[0] for p in points), tuple(p[1] for p in points), marker='x', s=50, color='green')
            pyplot.show()

        if self.properties["Mode"] == "end-to-end":
            points.append((gt_angle, gt_tilt, gt_resolution))
        return points

    def targets_normalized(self):
        """Same as cutter.targets() but normalized to sensible values."""

        return self.normalize(self.targets())

    def calculate_norms(self):
        """Calculated sensible values to normalize (image independent)."""

        if not self.norms_modified:
            return self.norms

        # normalize and save the head position and orientation points in the targets array
        # bounds for the headpoint normalization
        height_final = int(self.properties["Final"]["Height"])
        width_final  = int(self.properties["Final"]["Width"])
        center_offset= self.properties["CenterOffset"]
        min_resolution     = self.properties["MinResolution"]
        pixel_displacement = self.properties["MaxDisplacement"] / min_resolution

        self.max_angle_base = 3 * self.properties["MaxAngleRotation"]

        head_center = (width_final/2 - 0.5 + center_offset[0], height_final/2 - 0.5 + center_offset[1])

        head_left   = head_center[0] - (pixel_displacement + 0.5)
        head_right  = head_center[0] + (pixel_displacement + 0.5)
        head_top    = head_center[1] - (pixel_displacement + 0.5)
        head_bottom = head_center[1] + (pixel_displacement + 0.5)

        self.norms = []
        for i, dist in enumerate(self.properties["TargetDistances"]):
            pixel_dist = math.sqrt( dist[0]**2 + dist[1]**2 )  / min_resolution
            pixel_angle = math.atan2( dist[1], dist[0])
            max_angle = self.max_angle_base + math.degrees(pixel_angle)
            min_angle = -self.max_angle_base + math.degrees(pixel_angle)

            # 90 deg in there
            if min_angle <= 90.0 and max_angle >= 90.0:
                y_min = -1.0
            else: 
                y_min = -max(math.sin(math.radians(min_angle)), math.sin(math.radians(max_angle)))
            # -90 deg in there
            if min_angle <= -90.0 and max_angle >= -90.0:
                y_max = 1.0
            else: 
                y_max = -min(math.sin(math.radians(min_angle)), math.sin(math.radians(max_angle)))
            # 180 deg or -180 deg in there
            if (min_angle <= 180.0 and max_angle >= 180.0) or (min_angle <= -180.0 and max_angle >= -180.0):
                x_min = -1.0
            else: 
                x_min = min(math.cos(math.radians(min_angle)), math.cos(math.radians(max_angle)))
            # 0 deg in there
            if min_angle <= 0.0 and max_angle >= 0.0:
                x_max = 1.0
            else: 
                x_max = max(math.cos(math.radians(min_angle)), math.cos(math.radians(max_angle)))

            # calculated normalization regions
            p_left   = head_left   + x_min * pixel_dist
            p_right  = head_right  + x_max * pixel_dist
            p_top    = head_top    + y_min * pixel_dist
            p_bottom = head_bottom + y_max * pixel_dist

            self.norms.append( (p_left, p_right, p_top, p_bottom) )

        self.norms_modified = False
        return self.norms

    def normalize(self, targets):
        """Normalize the passed targets by the objects norms.

        targets = cutter.normalize(pixel_targets)

        where
        pixel_targets 	is a list of 2d tuples of targets in pixel values and
        targets 		is a list of 2d tuples of targets normalized to the image present."""

        if self.norms_modified:
            self.calculate_norms()

        if len(targets) != (len(self.norms) + (3 if self.properties["Mode"] == "end-to-end" else 0)):
            raise CutterException("The number of points for the target did not agree with the number of points in the cutter (by property 'TargetDistances'")

        normalized_points = []
        if self.show_images:
            pyplot.set_cmap(pyplot.gray())
            pyplot.imshow(self.image_final)
            pyplot.title("Final image with center in blue and normalization boxes in blue")
            pyplot.scatter(self.final_x, self.final_y, marker='x', s=50, color='blue')
#			pyplot.plot((0, widthFinal, widthFinal, newRight, self.final_left), (newTop, newBottom, newBottom, newTop, newTop), linestyle='solid',linewidth=1, color='black')

        for i, norm in enumerate(self.norms):
            p_left   = norm[0]
            p_right  = norm[1]
            p_top    = norm[2]
            p_bottom = norm[3]

            if self.show_images:
                # plot the box of the i-th point
                pyplot.plot(
                    (p_left, p_left, p_right, p_right, p_left), 
                    (p_top, p_bottom, p_bottom, p_top, p_top), linestyle='solid',linewidth=1, color='blue')

            # calculate the size of the box
            distance_left_right  = p_right - p_left
            distance_top_bottom  = p_bottom - p_top

            # normalize points
            px = (targets[i][0] - p_left - distance_left_right/2) / (distance_left_right/2) 
            py = (targets[i][1] - p_top  - distance_top_bottom/2) / (distance_top_bottom/2)

            normalized_points.append( (px, py) )

            if self.verbose:
                print ('normalized point ' + str(i) + ' to ' + str(normalized_points[i]))

        # position marker of the center in blue
        if self.show_images:
            pyplot.show()

        if self.properties["Mode"] == "end-to-end":
            normalized_points.append((self.final_angle / self.max_angle_base, 
                self.tilt / 90., 
                (self.resolution - self.properties["MinResolution"]) / (self.properties["MaxResolution"] - self.properties["MinResolution"]) * 2 - 1 ))
        return normalized_points

    def unnormalize(self, targets):
        """Inverse of normalization. 

        pixel_targets = cutter.unnormalize(targets)

        where 
        targets 		is a list of tuples with normalized x- and y-coordiantes and
        pixel_targets 	is a list of tuples with pixel targets."""

        if self.norms_modified:
            self.calculate_norms()

        if len(targets) != len(self.norms) + (1 if self.properties["Mode"] == "end-to-end" else 0):
            raise CutterException("The number of points for the target did not agree with the number of points in the cutter (by property 'TargetDistances'")

        unnormalized_points = []

        for i, norm in enumerate(self.norms):
            p_left   = norm[0]
            p_right  = norm[1]
            p_top    = norm[2]
            p_bottom = norm[3]

            # calculate the size of the box
            distance_left_right  = p_right - p_left
            distance_top_bottom  = p_bottom - p_top

            # normalize points			
            px = (targets[i][0] * (distance_left_right/2) + p_left + distance_left_right/2) 
            py = (targets[i][1] * (distance_top_bottom/2) + p_top  + distance_top_bottom/2)

            unnormalized_points.append( (px, py) )

            if self.verbose:
                print ('unnormalized point ' + str(i) + ' to ' + str(unnormalized_points[i]))

        if self.properties["Mode"] == "end-to-end":
            unnormalized_points.append((targets[-1][0] * self.max_angle_base, 
                targets[-1][1] * 90., 
                (targets[-1][2] + 1.)/2. * (self.properties["MaxResolution"] - self.properties["MinResolution"]) + self.properties["MinResolution"]))
        return unnormalized_points
    
class CutterException (BaseException):
    def __init__(self, message):
        self.s_message = message

    def message(self):
        return self.s_message

