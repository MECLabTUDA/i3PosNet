#!/usr/bin/python

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

import os
import json
import platform
import sys
import warnings
folder = os.path.normpath(os.path.realpath(sys.path[0]) + "/code")
sys.path.append(folder)

from argparse_helper import *
from argparse import ArgumentParser, FileType

argument_parser = ArgumentParser(
    description="Evaluate i3PosNet",
    fromfile_prefix_chars='@'
    )

add_instrument_options(argument_parser)

add_dataset_folder(argument_parser)

argument_parser.add_argument(
    '--image_name',
    choices=["drr", "xray"],
    default="drr",
    help="prefix of the image files in the dataset.")

add_samples(argument_parser, default=10)

add_interpolation(argument_parser)

argument_parser.add_argument(
    '--invert',
    action='store_true',
    default=False,
    help="Whether the intensities have to be inverted (true for i3PosNet's Dataset C, default: False)")

argument_parser.add_argument(
    '--model_dir',
    type=str,
    required=True,
    help="Path to the model folder. This folder is expected to have sub-folders `snapshots`, `hyperparam` and `history`.")
    
argument_parser.add_argument(
    '--model_prefix',
    type=str,
    default="i3PosNet_",
    help="Prefix of the model files to use (from `snapshots`, `hyperparam` and `history` folders).")

argument_parser.add_argument(
    '--model_postfix',
    type=str,
    default="",
    help="Postfix of the model file to use, this would be ``` for final model, `_best(<iteration_number>)` for best intermediate model (by validation loss) and `_<epoch>` for snapshots (default: ``).")

argument_parser.add_argument(
    '--model_timestamp',
    type=str,
    required=True,
    help="Timestamp of the model files to use (from `snapshots`, `hyperparam` and `history` folders).")

argument_parser.add_argument(
    '--iterations',
    type=unsigned,
    default=3,
    help="Number of to use for i3PosNet (default: 3).")
    
argument_parser.add_argument(
    '-o', '--out',
    type=FileType(w),
    required=True,
    help="The full output filename to use for results (json).")
    
add_plot_verbose(argument_parser)

add_range_list_subparser(argument_parser)

if __main__ == "__name__":

    # set the rng to defined settings
    # from keras_impl import seedsettings
    from predictorkeras import PredictorKeras as PredictorLib

    from cutter import Cutter

    cut = Cutter()
    
    # width and height of Large and Final images
    cut["Large"] = { "Width": 270, "Height": 270 }
    
    instument = args.instrument if hasattr(args, 'instrument') else 'unknown'
    # centerOffset (this is the position the object is placed at in ideal cases, it should be dependent on the object geometry)
    # screw: (-10, 0) depending on image size
    if instrument in ['robot', 'drill']:
        cut["CenterOffset"] = (10, 0)
    else:
        cut["CenterOffset"] = args.center_offset

    # maximal displacement of the headPoint in millimeters

    with open(folder + '/data/config.json') as fp:
        config = json.load(fp)

    cut["MinResolution"] = config["minResolution"]
    cut["InvertIntensities"] = False if hasattr(args, 'invert') else args.invert

    # input data directory
    cut["ImageDirectory"]  = args.data_dir + 'images/'
    cut["TrackedDirectory"] = args.data_dir + 'tracked/'
    parameter_dir = args.data_dir + 'parameters/'

    cut["Order"] = args.interpolation_order

    cut["Mode"] = 'end-to-end' if hasattr(args, 'end_to_end') and args.end_to_end else 'modular'
    if cut["Mode"] == "modular":
        # list of distances to move points on the main axis (this could be made to vectors, if the rotation of the object itself was to be considered (future work)
        # screw: [0, 1.5, 3, -1.5]
        if instrument == "screw":
            cut["TargetDistances"] = [(0., 0.), (1.5, 0.), (3., 0.), (-1.5, 0.), (0, 1.5), (0, -1.5)]
        # drill or robot: [0, 1.5, -3, -1.5]
        elif instrument in ["drill", "robot"]:
            cut["TargetDistances"] = [(0., 0.), (1.5, 0.), (-3., 0.), (-1.5, 0.), (0, 1.5), (0, -1.5)]
        else:
            cut["TargetDistances"] = args.custom_landmarks

        # number of outputs to be extracted from the image (x and y head coordinates = 2 targets)
    elif cut["Mode"] == "end-to-end":
        # list of distances to move points on the main axis (this could be made to vectors, if the rotation of the object itself was to be considered (future work)
        # screw: [0, 1.5, 3, -1.5]
        cut["TargetDistances"] = [(0., 0.)]


    # number of outputs to be extracted from the image (x and y head coordinates = 2 targets) plus angle, tilt and resolution
    target_number = cut.target_number()

    # cropped image dimensions
    imageDimensions = (cut["Large"]["Width"], cut["Large"]["Height"])

    PredictorLib.verbose = args.verbose
    PredictorLib.show_images = args.plot

    if args.mode == 'range':
        iterator = range(args.start, args.stop, args.step)
        if hasattr(args, 'exclude') and args.exclude is not None and len(args.exclude) > 0:
            iterator = exclusion(iterator, args.exclude)
            numImages = len(iterator)
        else:
            numImages = int(math.ceil((endIndex-startIndex)/indexStep))
    elif args.mode == 'list':
        iterator = args.list
        numImages = len(list)
    else:
        raise RuntimeError("Invalid mode")

    results = []
    print ("Evaluating network.")
    
    model_filename = args.model_dir + "/snapshots/" + args.prefix + args.timestamp + args.postfix
    
    if not os.path.isfile(model_filename):
        raise RuntimeError(f"The model file {model_filename} is not a valid file, please check model_dir, prefix, timestamp and postfix.")
    
    predictor = PredictorLib(model_filename)
    predictor.properties["patterns"]["drr"] = args.image_name + '{:08d}.dcm'
    predictor["TrainingDirectory"] = args.model_dir + "/"
        
    r = {
        "results": None,
        "parameters": None,
        "history": None
        }
    filename = predictor["HyperParametersDirectory"] + args.prefix + args.timestamp + ".json"
    if not os.path.isfile(filename):
        warnings.warn("Could not find the hyperparameters file, skipping...")
    else:
        with open(filename) as fp:
            r["parameters"] = json.load(fp)
        
    filename = predictor["HistoryDirectory"] + args.prefix + args.timestamp + "_hist.json"
    if not os.path.isfile(filename):
        warnings.warn("Could not find the history file, skipping...")
    else:
        with open(filename) as fp:
            r["history"] = json.load(fp)
        
    predictor.set_cutter(cut)

    r["results"] = predictor.evaluate(
            imageindices = iterator, 
            iterations   = args.iterations,
            repeats      = args.samples
            )

    results.append(r)

    del predictor

    json.dump(results, out)
