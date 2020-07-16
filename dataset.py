#!/usr/bin/env python3

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

import h5py
import math
import numpy as np
import random

import os
import json
from argparse import ArgumentParser
from datetime import datetime

import sys

folder = os.path.normpath(os.path.join(os.path.realpath(sys.path[0]), 'code'))
sys.path.append(folder)

from argparse_helper import (
    add_instrument_options,
    add_dataset_folder,
    add_samples,
    add_interpolation,
    add_plot_verbose,
    add_range_list_subparser,
    exclusion
)


argument_parser = ArgumentParser(
    description='Creation of hdf5 training dataset for i3PosNet',
    fromfile_prefix_chars='@'
)

add_instrument_options(argument_parser)
add_dataset_folder(argument_parser)
add_samples(argument_parser)
add_interpolation(argument_parser)

argument_parser.add_argument(
    '--hdf_dir',
    type=str,
    default=os.getcwd(),
    help='Path to the folder to store the hdf dataset. The filename will be generated automatically (default: current working directory).'
)

add_plot_verbose(argument_parser)
add_range_list_subparser(argument_parser)


def normal_clamped(mean, stddev):
    d = np.random.normal(0, stddev)
    d = min(3 * stddev, max(-3 * stddev, d))
    return mean + d


if __name__ == "__main__":
    from cutter import Cutter

    args = argument_parser.parse_args()

    cut = Cutter()

    instrument = args.instrument if hasattr(args, 'instrument') else 'unknown'
    # centerOffset (this is the position the object is placed at in ideal
    # cases, it should be dependent on the object geometry)
    # screw: (-10, 0) depending on image size
    if instrument in ['robot', 'drill']:
        cut['CenterOffset'] = (10, 0)
    else:
        cut['CenterOffset'] = args.center_offset

    # maximal displacement of the headPoint in millimeters

    with open(os.path.join(folder, '../data/config.json'), 'r') as fp:
        config = json.load(fp)

    cut['MinResolution'] = config['minResolution']
    cut['HDF5Directory'] = args.hdf_dir

    # input data directory
    cut['ImageDirectory'] = args.data_dir + 'images/'
    cut['TrackedDirectory'] = args.data_dir + 'tracked/'
    parameter_dir = args.data_dir + 'parameters/'

    cut['Order'] = args.interpolation_order
    cut['Mode'] = 'end-to-end' if hasattr(args, 'end_to_end') and args.end_to_end else 'modular'
    if cut['Mode'] == 'modular':
        # list of distances to move points on the main axis (this could be made to vectors, if the rotation of the object itself was to be considered (future work)
        # screw: [0, 1.5, 3, -1.5]
        if instrument == 'screw':
            cut['TargetDistances'] = [
                (0., 0.), (1.5, 0.), (3., 0.), (-1.5, 0.), (0, 1.5), (0, -1.5)
            ]
        # drill or robot: [0, 1.5, -3, -1.5]
        elif instrument in ['drill', 'robot']:
            cut['TargetDistances'] = [
                (0., 0.), (1.5, 0.), (-3., 0.), (-1.5, 0.), (0, 1.5), (0, -1.5)
            ]
        else:
            cut['TargetDistances'] = args.custom_landmarks
        # number of outputs to be extracted from the image
        # (x and y head coordinates = 2 targets)
    elif cut['Mode'] == 'end-to-end':
        # list of distances to move points on the main axis
        # (this could be made to vectors, if the rotation of the object itself
        # was to be considered (future work)
        # screw: [0, 1.5, 3, -1.5]
        cut['TargetDistances'] = [(0., 0.)]

    # number of outputs to be extracted from the image
    # (x and y head coordinates = 2 targets) plus angle, tilt and resolution
    target_number = cut.target_number()

    # cropped image dimensions
    imageDimensions = (cut['Large']['Width'], cut['Large']['Height'])

    # number of hdf5 samples per drr
    numCuts = args.num_samples

    # if len(strWidth) > 0: widthFinal = int(strWidth)
    # if len(strHeight) > 0: heightFinal = int(strHeight)

    cut.verbose = args.verbose
    cut.show_images = args.plot

    if args.mode == 'range':
        endIndex = args.stop
        startIndex = args.start
        indexStep = args.step
        iterator = range(args.start, args.stop, args.step)
        if hasattr(args, 'exclude') and args.exclude is not None and len(args.exclude) > 0:
            iterator = exclusion(iterator, args.exclude)
            numImages = len(iterator)
        else:
            numImages = int(math.ceil((endIndex - startIndex) / indexStep))
    elif args.mode == 'list':
        iterator = args.list
        numImages = len(list)
    else:
        raise RuntimeError('Invalid mode')

    # create images and targets arrays for the hdf5 file (crop each image twice)
    images = np.zeros((numImages * numCuts, 1, cut['Final']['Height'], cut['Final']['Width']), dtype='f8')
    targets = np.zeros((numImages * numCuts, target_number), dtype='f8')

    for i, index in enumerate(iterator):
        #save image path and robot head position
        trackedJsonPath = f'tracked{index:08d}.JSON'
        parameterJsonPath = f'parameter{index:08d}.JSON'
        # FIXME prefix can be "drr" or "xray"
        prefix = 'drr'
        imagePath = f'{prefix}{index:08d}.dcm'

        # load parameter-json file
        # with open(parameter_dir + parameterJsonPath) as parameter_file:
        #    json_data = json.load(parameter_file)
        #    x.append(json_data['Position']['X'])
        #    y.append(json_data['Position']['Y'])
        #    z.append(json_data['Position']['Z'])

        # load json and image file
        cut.load(dicom_file=imagePath, tracked_file=trackedJsonPath)

        # we add noise to the original position of the data
        for j in range(numCuts):
            # rotate image with normally distributed angle with sigma = 10 degrees
            normalDistAngle = np.random.normal(0, config['angleStddev'])
            cut.rotate(normalDistAngle)

            # generate random image center point in range -12 - + 12 from the current ceiled center point
            randomRadius = random.uniform(0, config['displacement'])
            randomPhi = random.uniform(0, 360) # in degrees

            images[i + j * numImages] = cut.cut_to((
                randomRadius * math.cos(math.radians(randomPhi)),
                randomRadius * math.sin(math.radians(randomPhi))
            ))

            t = cut.targets_normalized()
            tmp = []
            for t_ in t:
                for t__ in t_:
                    tmp.append(t__)
            targets[i + j * numImages] = tmp

        if i % 10 == 0:
            print(f'\rcompleted input image {i} of {numImages} images')

    print('Writing images and target information to file...')

    # write the images and their corresponding head positions to hdf5
    if args.mode == 'range':
        h5_filename = cut['HDF5Directory'] + f'{numCuts:02d}_{args.start:08d}-{args.stop:08d}_{args.step:d}'
    else:
        h5_filename = cut['HDF5Directory'] + 'list_' + str(datetime.now())
    if hasattr(args, 'end_to_end') and args.end_to_end:
        h5_filename += '_e2e'
    if args.interpolation_order != 3:
        h5_filename += f'_o{args.interpolation_order:d}'
    if config['displacement'] != 2.5:
        h5_filename += '_disp{:.1f}'.format(config['displacement'])
    if config['angleStddev'] != 10.0:
        h5_filename += '_aStd{:.1f}'.format(config['angleStddev'])
    h5_filename += '.h5'

    with h5py.File(h5_filename, 'w') as h5file:
        h5file.create_dataset('images', data=images)
        h5file.create_dataset('targets', data=targets)

    print("done.")
